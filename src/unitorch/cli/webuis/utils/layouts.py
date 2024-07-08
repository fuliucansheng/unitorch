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
            info=info,
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

    raise ValueError(f"Unknown element type: {dtype}")


def create_accordion(*elements, name=None, open=False):
    accordion = gr.Accordion(label=name, open=open)
    for element in elements:
        accordion.add_child(element)
    return accordion


def create_row(*elements, variant="panel"):
    row = gr.Row(variant=variant)
    for element in elements:
        row.add_child(element)
    return row


def create_column(*elements, variant="panel", scale=1):
    col = gr.Column(
        variant=variant,
        scale=scale,
    )
    for element in elements:
        col.add_child(element)
    return col


def create_group(*elements):
    group = gr.Group()
    for element in elements:
        group.add_child(element)
    return group


def create_tab(*elements, name=None):
    tab = gr.Tab(label=name)
    for element in elements:
        tab.add_child(element)
    return tab


def create_tabs(*elements):
    tabs = gr.Tabs()
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

    controlnets = [create_controlnet() for _ in range(num_controlnets)]
    tabs = [
        create_tab(controlnet[-1], name=f"Net {i}")
        for i, controlnet in enumerate(controlnets)
    ]
    layout = create_accordion(create_tabs(*tabs), name="ControlNets")
    return GenericOutputs(
        controlnets=[controlnet[0] for controlnet in controlnets], layout=layout
    )


def create_lora_layout(loras: Union[List[str], Dict[str, List[str]]]):
    def create_lora(loras: List[str]):
        eles = [
            create_element("button", label=lora, elem_classes=["group-lora-item"])
            for lora in loras
        ]
        layout = create_row(*eles)
        return eles, layout

    if isinstance(loras, list):
        eles, layout = create_lora(loras)
        layout = create_accordion(layout, name="LORA")
        return eles, layout
    if isinstance(loras, dict):
        groups = [
            create_tab(create_lora(lora), name=name) for name, lora in loras.items()
        ]
        eles = [group[0] for group in groups]
        tabs = [group[1] for group in groups]
        layout = create_tabs(*tabs)
        layout = create_accordion(layout, name="LORA")
        return eles, layout
    raise ValueError(f"Unsupported lora type: {type(loras)}")


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
