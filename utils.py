from groundingdino.util.inference import load_model, predict, annotate
from groundingdino.util import box_ops
from PIL import Image, ImageEnhance
from typing import Tuple
from diffusers import StableDiffusionInpaintPipeline
from transformers import SamModel, SamProcessor
from perceiver.model.vision import optical_flow  # noqa: F401
from transformers import pipeline
import groundingdino.datasets.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import requests
import re
import copy


class Prompts:
    def __init__(
        self, search_prompt, replace_prompt=None, negative_prompt="low quality"
    ) -> None:
        self.replace_prompt = replace_prompt
        self.negative_prompt = negative_prompt
        self.search_prompt = search_prompt


class GenAiModels:
    def __init__(self) -> None:
        gd_model = load_model("utils.py", "weights/groundingdino_swinb_cogcoor.pth")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        ).to(device)
        self.optical_flow_pipeline = pipeline(
            "optical-flow", model="krasserm/perceiver-io-optical-flow", device="cuda:0"
        )
        self.processor = processor
        self.sam_model = sam_model
        self.gd_model = gd_model
        self.sd_pipe = sd_pipe

    def detect(
        self, image, tensor_image, text_prompt, box_threshold=0.3, text_threshold=0.3
    ):
        boxes, logits, phrases = predict(
            model=self.gd_model,
            image=tensor_image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        # print(boxes, logits)
        boxes_list = [boxes[b] for b in range(len(phrases)) if phrases[b] != ""]
        if len(boxes_list) == 0:
            return image, [[]]
        boxes = torch.stack(boxes_list)
        phrases = [phrases[b] for b in range(len(phrases)) if phrases[b] != ""]
        annotated_frame = annotate(
            image_source=image, boxes=boxes, logits=logits, phrases=phrases
        )
        # BGR to RGB
        annotated_frame = annotated_frame[..., ::-1]

        return annotated_frame, boxes

    def generate_image(self, image, mask, prompt, negative_prompt, seed, device="cuda"):
        # resize for inpainting
        w, h = image.size
        in_image = image.resize((1024, 1024))
        in_mask = mask.resize((1024, 1024))

        generator = torch.Generator(device).manual_seed(seed)

        result = self.sd_pipe(
            image=in_image,
            mask_image=in_mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=1,
            generator=generator,
        )
        result = result.images[0]

        return result.resize((w, h))

    def search_replace(self, image, prompt_obj: Prompts, seed=10, device="cuda"):
        generated_image, _ = self.search_replace_with_mask(
            image, prompt_obj, seed, device
        )
        return generated_image

    def prompt_segmentation(self, image, prompt_obj: Prompts, seed=1, device="cuda"):
        raw_image, array_image, tensor_image = transform_image(image, 0)
        annotated_frame, detected_boxes = self.detect(
            array_image,
            tensor_image,
            prompt_obj.search_prompt,
            box_threshold=0.3,
            text_threshold=0.3,
        )

        H, W, _ = array_image.shape
        if len(detected_boxes[0]) == 0:
            return image, image
        boxes = [
            (box_ops.box_cxcywh_to_xyxy(detected_boxes) * torch.Tensor([W, H, W, H]))
            .cpu()
            .tolist()
        ]
        if len(boxes[0]) == 0:
            return image, image
        inputs = self.processor(raw_image, input_boxes=boxes, return_tensors="pt").to(
            device
        )
        image_embeddings = self.sam_model.get_image_embeddings(inputs["pixel_values"])

        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})

        with torch.no_grad():
            outputs = self.sam_model(**inputs, multimask_output=True)

            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
        scores = outputs.iou_scores

        # show_masks_on_image(raw_image, masks[0], scores)

        # merge all masks into a single final mask

        final_mask = None

        if len(masks[0]) == 1:
            final_mask = masks[0][0][
                np.argmax(scores.cpu().numpy()[0][0])
            ]  # .numpy().astype(int)
        else:
            for i in range(len(masks[0]) - 1):
                if final_mask is None:
                    final_mask = np.bitwise_or(masks[0][i][0], masks[0][i + 1][0])
                else:
                    final_mask = np.bitwise_or(final_mask, masks[0][i + 1][0])

        annotated_frame_with_mask, _ = draw_mask(final_mask, array_image, False)

        return Image.fromarray(annotated_frame_with_mask), Image.fromarray(
            annotated_frame
        )

    def prompt_segmentation_list(
        self, image_list, prompt_obj: Prompts, seed=1, device="cuda"
    ):
        generated_list = []
        for image_tmp in image_list:
            if type(image_tmp) == str:
                image_tmp = load_image_func(image_tmp, 512)
            result_tmp, _ = self.prompt_segmentation(image_tmp, prompt_obj, 3)
            generated_list.append(result_tmp)

        return generated_list

    def remove_from_image(self, image, prompt_obj, seed=1, device="cuda"):
        prompt_obj.replace_prompt = "background"
        return self.search_replace(image, prompt_obj, seed, device)

    def search_replace_with_mask(
        self, image, prompt_obj: Prompts, seed=1, device="cuda"
    ):
        raw_image, array_image, tensor_image = transform_image(image, 0)
        annotated_frame, detected_boxes = self.detect(
            array_image,
            tensor_image,
            prompt_obj.search_prompt,
        )

        H, W, _ = array_image.shape
        boxes = [
            (box_ops.box_cxcywh_to_xyxy(detected_boxes) * torch.Tensor([W, H, W, H]))
            .cpu()
            .tolist()
        ]

        with torch.no_grad():
            inputs = self.processor(
                raw_image, input_boxes=boxes, return_tensors="pt"
            ).to(device)
            image_embeddings = self.sam_model.get_image_embeddings(
                inputs["pixel_values"]
            )

        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})

        with torch.no_grad():
            outputs = self.sam_model(**inputs, multimask_output=True)

            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
        scores = outputs.iou_scores

        # show_masks_on_image(raw_image, masks[0], scores)

        # merge all masks into a single final mask

        final_mask = None

        if len(masks[0]) == 1:
            final_mask = masks[0][0][
                np.argmax(scores.cpu().numpy()[0][0])
            ]  # .numpy().astype(int)
        else:
            for i in range(len(masks[0]) - 1):
                if final_mask is None:
                    final_mask = np.bitwise_or(masks[0][i][0], masks[0][i + 1][0])
                else:
                    final_mask = np.bitwise_or(final_mask, masks[0][i + 1][0])

        annotated_frame_with_mask = draw_mask(final_mask, array_image)
        # Image.fromarray(annotated_frame_with_mask)

        # create mask images

        mask = final_mask.cpu().numpy()

        # mask_image.cpu().numpy() * 255

        mask = dilate_mask(mask.astype(np.uint8) * 255, dilate_factor=20)
        # mask = mask.cpu().numpy()

        inverted_mask = 255 - mask  # .astype(np.uint8)

        image_source_pil = Image.fromarray(array_image)  # .resize((1024, 1024))
        image_mask_pil = Image.fromarray(mask)
        inverted_image_mask_pil = Image.fromarray(inverted_mask)

        with torch.no_grad():
            generated_image = self.generate_image(
                image=image_source_pil,
                mask=image_mask_pil,
                prompt=prompt_obj.replace_prompt,
                negative_prompt=prompt_obj.negative_prompt,
                seed=seed,
            )

        return generated_image, image_mask_pil

    def search_replace_list(
        self,
        images_list,
        prompt_object: Prompts,
        temporal_smoothing=True,
        fancy_flow=False,
        seed=3,
    ):
        generated_list = []
        generated_mask = []
        for k1, image_str in enumerate(images_list):
            contrast = 1.8 if image_str.find("images") >= 0 else -1
            if k1 > 0 and temporal_smoothing:
                im1 = load_image_func(images_list[k1 - 1], 512, contrast)
                im2 = load_image_func(images_list[k1], 512, contrast)
                if fancy_flow:
                    flow = self.optical_flow_pipeline((im1, im2), render=False)
                else:
                    flow, _ = optical_flow_classic(im1, im2)
            else:
                im2 = load_image_func(images_list[k1], 512, contrast)

            img_tmp, mask = self.search_replace_with_mask(im2, prompt_object, seed)

            if k1 > 0 and temporal_smoothing:
                img_tmp = paste_optical_flow_with_mask_rev(
                    generated_list[k1 - 1],
                    img_tmp,
                    flow,
                    np.array(generated_mask[k1 - 1]),
                )
            generated_list.append(img_tmp)
            generated_mask.append(mask)

        return generated_list

    def search_replace_image_list(
        self, images_list, prompt_object: Prompts, temporal_smoothing=True, seed=3
    ):
        generated_list = []
        generated_mask = []
        for k1, image_str in enumerate(images_list):
            if k1 > 0 and temporal_smoothing:
                im1 = images_list[k1 - 1]
                im2 = images_list[k1]
                flow, _ = optical_flow_classic(im1, im2)
            else:
                im2 = images_list[k1]

            img_tmp, mask = self.search_replace_with_mask(im2, prompt_object, seed)
            if k1 > 0 and temporal_smoothing:
                img_tmp = paste_optical_flow_with_mask_rev(
                    generated_list[k1 - 1],
                    img_tmp,
                    flow,
                    np.array(generated_mask[k1 - 1]),
                )
            generated_list.append(img_tmp)
            generated_mask.append(mask)

        return generated_list


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            w,
            h,
            edgecolor="green",
            facecolor=(0, 0, 0, 0),
            lw=2,
        )
    )


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.show()


def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis("on")
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if scores.shape[0] == 1:
        scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = mask.cpu().detach()
        axes[i].imshow(np.array(raw_image))
        show_mask(mask, axes[i])
        axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
        axes[i].axis("off")
    plt.show()


def load_image(image_url: str, q) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    H, W = image_source.size
    image_source = image_source.resize((int((1 - q) * H), int((1 - q) * W)))
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image_source, image, image_transformed


def dilate_mask(mask, dilate_factor):
    # mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask, np.ones((dilate_factor, dilate_factor), np.uint8), iterations=1
    )
    return mask


def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:  # 125, 249, 255
        color = np.array([124 / 255, 252 / 255, 0 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray(
        ((mask_image.cpu().numpy() * 255)).astype(np.uint8)
    ).convert("RGBA")
    composite = Image.alpha_composite(annotated_frame_pil, mask_image_pil)
    return np.array(composite), composite


def transform_image(image_ori, q):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = copy.deepcopy(image_ori)
    H, W = image_source.size
    # q = 1024/W
    # print(f"########## q is  {q}")
    image_source = image_source.resize((int((1 - q) * H), int((1 - q) * W)))
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image_source, image, image_transformed


def natural_sort(l):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    return sorted(l, key=lambda key: [convert(c) for c in re.split("([0-9]+)", key)])


def load_image_func(image_str: str, size=None, contrast=-1):
    contrast = 1.7 if image_str.find("images") >= 0 else -1
    # contrast = 1.8 if image_str.find("data") >= 0 else -1
    if image_str.startswith("http"):
        image_tmp = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image_tmp = Image.open(image_str).convert("RGB")
    if contrast > 0:
        image_tmp = ImageEnhance.Contrast(image_tmp).enhance(contrast)
    if size is not None:
        w, h = image_tmp.size
        q = size / w
        image_tmp = image_tmp.resize((int(w * q), int(h * q)))

    return image_tmp


# Calculates dense optical flow by Farneback method
def optical_flow_classic(im1, im2, return_flow=False):
    im1_gray = np.array(im1.convert("L"))
    im2_gray = np.array(im2.convert("L"))
    flow = cv2.calcOpticalFlowFarneback(
        im1_gray, im2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    if not return_flow:
        return flow, None
    mask = np.zeros_like(im1)

    # Sets image saturation to maximum
    mask[..., 1] = 255

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Sets image hue according to the optical flow
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    return flow, Image.fromarray(rgb)


def paste_optical_flow(im1, im2, flow):
    im3 = im2.copy()
    for k1 in range(flow.shape[0]):
        for k2 in range(flow.shape[1]):
            dx = min(max(int(k1 + flow[k1, k2, 0]), 0), flow.shape[0] - 1)
            dy = min(max(int(k2 + flow[k1, k2, 1]), 0), flow.shape[1] - 1)
            im3[dx, dy, :] = im1[k1, k2, :]

    return Image.fromarray(im3[:, :, [2, 1, 0]])


def paste_optical_flow_with_mask(im1, im2, flow, mask):
    im1_ = np.array(im1)
    im3_ = np.array(im2)
    alpha = 0.1
    for k1 in range(flow.shape[0]):
        for k2 in range(flow.shape[1]):
            dx = min(max(int(k2 + flow[k1, k2, 0]), 0), flow.shape[1] - 1)
            dy = min(max(int(k1 + flow[k1, k2, 1]), 0), flow.shape[0] - 1)

            if mask[dy, dx] > 0:
                im3_[dy, dx, :] = im1_[k1, k2, :]
                im3_[dy, dx, :] = (
                    im1_[k1, k2, :] * alpha + im3_[dy, dx, :] * (1 - alpha)
                ).astype("uint8")

    im3_blurred = cv2.GaussianBlur(im3_, (7, 7), 0)
    for k1 in range(3):
        im3_[..., k1] = np.where(mask != 0, im3_blurred[..., k1], im3_[..., k1])

    return Image.fromarray(im3_)


def paste_optical_flow_with_mask_rev(im1, im2, flow, mask):
    im1_ = np.array(im1)
    im3_ = np.array(im2)
    alpha = 0.9
    for k1 in range(flow.shape[0]):
        for k2 in range(flow.shape[1]):
            if mask[k1, k2] == 0:
                continue
            dx = min(max(int(k2 + flow[k1, k2, 0]), 0), flow.shape[1] - 1)
            dy = min(max(int(k1 + flow[k1, k2, 1]), 0), flow.shape[0] - 1)
            # dx = int(k2 + flow[k1,k2,0])
            # dy = int(k1 + flow[k1,k2,1])

            if mask[k1, k2] > 0:
                im3_[dy, dx, :] = (
                    im1_[k1, k2, :] * alpha + im3_[dy, dx, :] * (1 - alpha)
                ).astype("uint8")

    im3_blurred = cv2.GaussianBlur(im3_, (5, 5), 0)
    for k1 in range(3):
        im3_[..., k1] = np.where(mask != 0, im3_blurred[..., k1], im3_[..., k1])

    return Image.fromarray(im3_)


def plot_images(list_images):
    fig, ax = plt.subplots(
        1,
        len(list_images),
        figsize=(list_images[0].size[1] / 4, list_images[0].size[0] / 4 * 2),
    )

    for k1, image in enumerate(list_images):
        ax[k1].imshow(image)
        ax[k1].axis("off")


batch_size = 1
modelname = "groundingdino"
backbone = "swin_B_384_22k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_text_len = 256
text_encoder_type = "bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True
