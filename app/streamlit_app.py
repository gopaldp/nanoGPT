import streamlit as st
import torch
import sys
import os

sys.path.append('src')

from llm_project.models.gpt import GPT
import tiktoken


@st.cache_resource
def load_model():
    """Load model once and cache it"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('models/checkpoints/ckpt.pt',
                            map_location=device,
                            weights_only=False)
    model = GPT(checkpoint['model_args'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, tiktoken.get_encoding("gpt2"), device


def main():
    st.set_page_config(page_title="Shakespeare GPT", page_icon="🎭", layout="wide")

    st.title("🎭 Shakespeare GPT - Your AI Playwright")
    st.markdown("*Generate Shakespearean text with your custom-trained LLM*")

    # Sidebar for settings
    with st.sidebar:
        st.header("Generation Settings")
        temperature = st.slider("Temperature (creativity)", 0.1, 2.0, 0.8, 0.1)
        max_tokens = st.slider("Max tokens", 50, 500, 150, 50)
        top_k = st.slider("Top-k sampling", 10, 300, 200, 10)

        st.markdown("---")
        st.markdown("### Quick Prompts")
        if st.button("ROMEO:"):
            st.session_state.prompt = "ROMEO:"
        if st.button("JULIET:"):
            st.session_state.prompt = "JULIET:"
        if st.button("HAMLET:"):
            st.session_state.prompt = "HAMLET:"

    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer, device = load_model()

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Input")
        prompt = st.text_area("Enter your prompt:",
                              value=st.session_state.get('prompt', 'ROMEO:'),
                              height=150,
                              key='prompt_input')

        generate_btn = st.button("✨ Generate Text", type="primary")

    with col2:
        st.subheader("🎭 Generated Output")
        output_placeholder = st.empty()

    if generate_btn and prompt:
        with st.spinner("Generating..."):
            # Encode prompt
            start_ids = tokenizer.encode(prompt)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

            # Generate
            with torch.no_grad():
                generated = model.generate(x, max_new_tokens=max_tokens,
                                           temperature=temperature, top_k=top_k)

            # Decode
            generated_text = tokenizer.decode(generated[0].tolist())

            # Display
            with col2:
                st.text_area("Generated text:", generated_text, height=300)
                st.success("✅ Generation complete!")

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using PyTorch & Streamlit")


if __name__ == "__main__":
    main()
