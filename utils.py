import base64
import html
from pathlib import Path

import streamlit as st


def load_css(path: str) -> None:
    css = Path(path).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def icon_data_uri(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def icon_label(icon_uri: str, text: str, tag: str = "h3", size: int = 18, class_name: str = "") -> str:
    klass = f' class="{class_name}"' if class_name else ""
    return (
        f"<{tag}{klass}><img src=\"{icon_uri}\" width=\"{size}\" height=\"{size}\" "
        f"style=\"vertical-align:middle;margin-right:8px;\"/>{html.escape(text)}</{tag}>"
    )


def sanitize_text(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")
