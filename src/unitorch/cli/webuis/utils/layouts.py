# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import gradio as gr
from unitorch.models import GenericOutputs


def create_element(
    dtype,
    label,
    default=None,
    values=[],
    min_value=None,
    max_value=None,
    step=None,
    scale=None,
    multiselect=None,
    info=None,
    interactive=None,
    variant=None,
    lines=1,
    placeholder=None,
    show_label=True,
    elem_id=None,
    elem_classes=None,
    link=None,
):
    if dtype == "text":
        return gr.Textbox(
            value=default,
            label=label,
            scale=scale,
            info=info,
            interactive=interactive,
            lines=lines,
            placeholder=placeholder,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "number":
        return gr.Number(
            value=default,
            label=label,
            scale=scale,
            info=info,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "slider":
        return gr.Slider(
            value=default,
            label=label,
            minimum=min_value,
            maximum=max_value,
            step=step,
            scale=scale,
            info=info,
            interactive=interactive,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "checkbox":
        return gr.Checkbox(
            value=default,
            label=label,
            scale=scale,
            info=info,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "checkboxgroup":
        return gr.CheckboxGroup(
            value=default,
            label=label,
            choices=values,
            scale=scale,
            info=info,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "radio":
        return gr.Radio(
            value=default,
            label=label,
            choices=values,
            scale=scale,
            info=info,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "dropdown":
        return gr.Dropdown(
            value=default,
            label=label,
            choices=values,
            scale=scale,
            multiselect=multiselect,
            info=info,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "dataframe":
        return gr.Dataframe(
            value=default,
            label=label,
            scale=scale,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
            datatype="markdown",
            wrap=True,
            height=1000,
        )

    if dtype == "image":
        return gr.Image(
            type="pil",
            value=default,
            label=label,
            scale=scale,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "anno_image":
        return gr.AnnotatedImage(
            value=default,
            label=label,
            scale=scale,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "image_editor":
        return gr.ImageEditor(
            type="pil",
            value=default,
            label=label,
            scale=scale,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
            eraser=gr.Eraser(),
            brush=gr.Brush(),
            canvas_size=(1024, 1024),
        )

    if dtype == "audio":
        return gr.Audio(
            label=label,
            scale=scale,
            info=info,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "video":
        return gr.Video(
            label=label,
            scale=scale,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "file":
        return gr.File(
            label=label,
            scale=scale,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "gallery":
        return gr.Gallery(
            label=label,
            scale=scale,
            interactive=interactive,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "button":
        interactive = True if interactive is None else interactive
        variant = "primary" if variant is None else variant
        return gr.Button(
            value=label,
            scale=scale,
            interactive=interactive,
            variant=variant,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "markdown":
        return gr.Markdown(
            value=label,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "download_button":
        return gr.DownloadButton(
            label=label,
            value=link,
            scale=scale,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    if dtype == "html":
        return gr.HTML(
            value=default,
            label=label,
            show_label=show_label,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    raise ValueError(f"Unknown element type: {dtype}")


def create_accordion(
    *elements, name=None, open=False, elem_id=None, elem_classes="ut-bg-transparent"
):
    accordion = gr.Accordion(
        label=name,
        open=open,
        elem_id=elem_id,
        elem_classes=elem_classes,
    )
    for element in elements:
        accordion.add_child(element)
    return accordion


def create_row(
    *elements, variant="panel", elem_id=None, elem_classes="ut-bg-transparent"
):
    row = gr.Row(
        variant=variant,
        equal_height=True,
        elem_id=elem_id,
        elem_classes=elem_classes,
    )
    for element in elements:
        row.add_child(element)
    return row


def create_column(
    *elements, variant="panel", scale=1, elem_id=None, elem_classes="ut-bg-transparent"
):
    col = gr.Column(
        variant=variant,
        scale=scale,
        elem_id=elem_id,
        elem_classes=elem_classes,
    )
    for element in elements:
        col.add_child(element)
    return col


def create_flex_layout(
    *eles,
    num_per_row=2,
    do_padding=False,
    elem_id=None,
    elem_classes="ut-bg-transparent",
    elem_place_holder_classes="ut-bg-transparent ut-place-holder-block",
):
    if do_padding:
        nums = num_per_row - len(eles) % num_per_row
        eles = list(eles) + [
            create_element(
                "markdown", "<div></div>", elem_classes=elem_place_holder_classes
            )
            for _ in range(nums)
        ]
    eles = [
        create_column(
            ele,
            elem_classes="ut-bg-transparent ut-0-margin-padding",
        )
        for ele in eles
    ]
    rows = [
        create_row(
            *eles[i : i + num_per_row],
            elem_classes="ut-bg-transparent ut-0-margin-padding",
        )
        for i in range(0, len(eles), num_per_row)
    ]
    return create_column(*rows, elem_id=elem_id, elem_classes=elem_classes)


def create_group(*elements, elem_id=None, elem_classes="ut-bg-transparent"):
    group = gr.Group(elem_id=elem_id, elem_classes=elem_classes)
    for element in elements:
        group.add_child(element)
    return group


def create_tab(*elements, name=None, elem_id=None, elem_classes="ut-bg-transparent"):
    tab = gr.Tab(label=name, elem_id=elem_id, elem_classes=elem_classes)
    for element in elements:
        tab.add_child(element)
    return tab


def create_tabs(*elements, elem_id=None, elem_classes="ut-bg-transparent"):
    tabs = gr.Tabs(elem_id=elem_id, elem_classes=elem_classes)
    for element in elements:
        tabs.add_child(element)
    return tabs


def create_blocks(*layouts):
    blocks = gr.Blocks()
    elements = list(layouts)
    while len(elements) > 0:
        element = elements.pop(0)
        if hasattr(element, "children"):
            elements += element.children
        blocks.blocks[element._id] = element

    for layout in layouts:
        blocks.add_child(layout)

    return blocks


# create some layouts


def create_pretrain_layout(pretrained_names: List[str], default_name: str):
    name = create_element(
        "dropdown",
        "Checkpoint",
        default=default_name,
        values=pretrained_names,
    )
    status = create_element("text", "Status", default="Stopped", interactive=False)
    start = create_element("button", "Start", variant="primary")
    stop = create_element("button", "Stop", variant="stop")
    layout = create_row(name, status, start, stop)
    return GenericOutputs(
        name=name,
        status=status,
        start=start,
        stop=stop,
        layout=layout,
    )


def create_controlnet_layout(
    controlnets: List[str], processes: List[str], num_controlnets: Optional[int] = 5
):
    def create_controlnet():
        input_image = create_element("image", "Input Image")
        output_image = create_element("image", "Output Image")
        checkpoint = create_element("dropdown", "Checkpoint", values=controlnets)
        guidance_scale = create_element(
            "slider", "Guidance Scale", min_value=0, max_value=2, step=0.1
        )
        process = create_element("radio", "Process", values=processes, default=None)
        layout = create_column(
            create_row(input_image, output_image),
            create_row(checkpoint, guidance_scale),
            create_row(process),
        )
        return (
            GenericOutputs(
                input_image=input_image,
                output_image=output_image,
                checkpoint=checkpoint,
                guidance_scale=guidance_scale,
                process=process,
            ),
            layout,
        )

    controlnet_groups = [create_controlnet() for _ in range(num_controlnets)]
    tabs = [
        create_tab(controlnet_group[-1], name=f"Net {i}")
        for i, controlnet_group in enumerate(controlnet_groups)
    ]
    layout = create_accordion(create_tabs(*tabs), name="ControlNets")
    return GenericOutputs(
        controlnets=[controlnet_group[0] for controlnet_group in controlnet_groups],
        layout=layout,
    )


def create_adapter_layout(
    adapters: List[str], processes: List[str], num_adapters: Optional[int] = 5
):
    def create_adapter():
        input_image = create_element("image", "Input Image")
        output_image = create_element("image", "Output Image")
        checkpoint = create_element("dropdown", "Checkpoint", values=adapters)
        guidance_scale = create_element(
            "slider", "Guidance Scale", min_value=0, max_value=2, step=0.1
        )
        process = create_element("radio", "Process", values=processes, default=None)
        layout = create_column(
            create_row(input_image, output_image),
            create_row(checkpoint, guidance_scale),
            create_row(process),
        )
        return (
            GenericOutputs(
                input_image=input_image,
                output_image=output_image,
                checkpoint=checkpoint,
                guidance_scale=guidance_scale,
                process=process,
            ),
            layout,
        )

    adapter_groups = [create_adapter() for _ in range(num_adapters)]
    tabs = [
        create_tab(controlnet_group[-1], name=f"Net {i}")
        for i, controlnet_group in enumerate(adapter_groups)
    ]
    layout = create_accordion(create_tabs(*tabs), name="Adapters")
    return GenericOutputs(
        adapters=[adapter_group[0] for adapter_group in adapter_groups],
        layout=layout,
    )


def create_lora_layout(
    loras: Union[List[str], Dict[str, List[str]]], num_loras: Optional[int] = 8
):
    def create_lora():
        checkpoint = create_element("dropdown", "Checkpoint", values=loras)
        text = create_element("markdown", "Placeholder README Text For LORA Checkpoint")
        weight = create_element(
            "slider", "Weight", default=1.0, min_value=0, max_value=3, step=0.1
        )
        alpha = create_element(
            "slider", "Alpha", default=32, min_value=0, max_value=128, step=1
        )
        url = create_element("text", "URL")
        file = create_element("file", "File")
        layout = create_column(
            create_accordion(text, name="Notes on Usage"),
            create_row(checkpoint, create_column(weight, alpha)),
            create_row(url),
            create_row(file),
        )
        return (
            GenericOutputs(
                checkpoint=checkpoint,
                text=text,
                weight=weight,
                alpha=alpha,
                url=url,
                file=file,
            ),
            layout,
        )

    lora_groups = [create_lora() for _ in range(num_loras)]
    tabs = [
        create_tab(lora_group[-1], name=f"LORA {i}")
        for i, lora_group in enumerate(lora_groups)
    ]
    layout = create_accordion(create_tabs(*tabs), name="LORAs")
    return GenericOutputs(
        loras=[lora_group[0] for lora_group in lora_groups], layout=layout
    )


def create_freeu_layout():
    s1 = create_element(
        "slider", "S1", default=0.9, min_value=0, max_value=10, step=0.1
    )
    s2 = create_element(
        "slider", "S2", default=0.2, min_value=0, max_value=10, step=0.1
    )
    b1 = create_element(
        "slider", "B1", default=1.2, min_value=0, max_value=10, step=0.1
    )
    b2 = create_element(
        "slider", "B2", default=1.4, min_value=0, max_value=10, step=0.1
    )
    layout = create_accordion(
        create_row(s1, s2), create_row(b1, b2), name="FreeU Params"
    )
    return GenericOutputs(s1=s1, s2=s2, b1=b1, b2=b2, layout=layout)
