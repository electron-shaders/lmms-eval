"""vLLM's single-worker OpenAI server with vanilla RAG's native clip loader.

Use this module instead of ``vllm.entrypoints.openai.api_server`` when serving
the vanilla RAG wrapper. Registration runs in the API process, where vLLM decodes
multimodal inputs; model workers receive the usual native video tensors.
"""

from __future__ import annotations


def main():
    from vllm.entrypoints.launchers.api_server.entry import main as serve

    from .native_video import register_video_loader

    register_video_loader()
    serve()


if __name__ == "__main__":
    main()
