import streamlit as st
import requests

st.set_page_config(page_title="Multimodal Search", layout="wide")
st.title("nuScenes Cross-Modal Search")

mode = st.radio("Mode", ["Text", "Image"], horizontal=True)
top_k = st.slider("Top K", 1, 20, 6)
expr = st.text_input("Milvus filter (optional)", placeholder="is_night == true")

if mode == "Text":
    text = st.text_input("Query", "a car driving at night")
    if st.button("Search") and text:
        r = requests.post(
            "http://localhost:8080/search/text",
            data={"text": text, "top_k": top_k, "expr": expr or ""},
        ).json()
        cols = st.columns(3)
        for i, hit in enumerate(r["results"]):
            with cols[i % 3]:
                st.image(hit["image_url"], use_container_width=True)
                st.caption(f"{hit['scene']} | score={hit['score']:.3f}")
else:
    upl = st.file_uploader("Upload image", type=["jpg", "png"])
    if upl and st.button("Search"):
        r = requests.post(
            "http://localhost:8080/search/image",
            files={"file": upl.getvalue()},
            data={"top_k": top_k, "expr": expr or ""},
        ).json()
        cols = st.columns(3)
        for i, hit in enumerate(r["results"]):
            with cols[i % 3]:
                st.image(hit["image_url"], use_container_width=True)
                st.caption(f"{hit['scene']} | score={hit['score']:.3f}")
