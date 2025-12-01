# Dual‑Model GGUF Chatbot

## How to run the app
```bash
cd chatbot
python app.py
```

## What the app does
- Loads **two** GGUF models (e.g., an online base model and a custom‑trained model).
- Accepts a user‑provided topic and lets the two models converse back‑and‑forth, showcasing their differing behaviours.
- Perfect for demos such as a mock presidential debate, product comparison, or any scenario where you want two personalities to discuss a subject.

## Example usage
Enter the paths to the two GGUF files, a topic like *"What should the next president prioritize?"*, and choose the number of exchange turns. The interface will display a formatted dialogue between **Model 1** and **Model 2**.
