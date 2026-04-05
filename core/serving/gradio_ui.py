"""
Interface Gradio — test manuel du modèle de détection de fraude.

Accessible sur http://ui.fraud.local (même pod que l'API FastAPI,
monté sur le port 7860).

Lancement :
    python -m core.serving.gradio_ui
"""

import os

import gradio as gr
import httpx

API_URL = os.environ.get("SERVING_API_URL", "http://localhost:8000")


def predict(
    amount: float,
    frequency_1h: int,
    frequency_24h: int,
    latitude: float,
    longitude: float,
    transaction_type: str,
    device_type: str,
) -> tuple[str, str, str]:
    """
    Appelle POST /predict et retourne (score, verdict, détails JSON).
    """
    payload = {
        "transaction_id":  f"gradio-{os.urandom(4).hex()}",
        "amount":          amount,
        "frequency_1h":    frequency_1h,
        "frequency_24h":   frequency_24h,
        "latitude":        latitude,
        "longitude":       longitude,
        "transaction_type": transaction_type,
        "device_type":     device_type,
    }

    try:
        response = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPError as exc:
        return "Erreur", "API indisponible", str(exc)

    score   = result["risk_score"]
    verdict = "FRAUDE" if result["is_fraud"] else "Légitime"
    color   = "🔴" if result["is_fraud"] else "🟢"
    details = (
        f"Score de risque : {score:.4f}\n"
        f"Seuil           : {result['threshold']}\n"
        f"Version modèle  : {result.get('model_version', 'N/A')}"
    )
    return f"{score:.4f}", f"{color} {verdict}", details


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Fraud Detection — Test UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Fraud Detection — Test manuel")
        gr.Markdown("Renseigne les caractéristiques d'une transaction pour obtenir son score de risque.")

        with gr.Row():
            with gr.Column():
                amount        = gr.Number(label="Montant (€)", value=150.0, minimum=0.01)
                frequency_1h  = gr.Slider(0, 50,  value=2,  step=1, label="Transactions / 1h")
                frequency_24h = gr.Slider(0, 200, value=10, step=1, label="Transactions / 24h")
                latitude      = gr.Number(label="Latitude",  value=48.8566)
                longitude     = gr.Number(label="Longitude", value=2.3522)
                tx_type       = gr.Dropdown(
                    ["online", "pos", "atm"], value="online", label="Type de transaction"
                )
                device        = gr.Dropdown(
                    ["mobile", "desktop", "unknown"], value="mobile", label="Device"
                )
                submit_btn    = gr.Button("Analyser", variant="primary")

            with gr.Column():
                score_out   = gr.Textbox(label="Score de risque")
                verdict_out = gr.Textbox(label="Verdict")
                details_out = gr.Textbox(label="Détails", lines=4)

        submit_btn.click(
            fn=predict,
            inputs=[amount, frequency_1h, frequency_24h, latitude, longitude, tx_type, device],
            outputs=[score_out, verdict_out, details_out],
        )

        gr.Markdown("---")
        gr.Markdown(
            "**Exemples de transactions suspectes** : montant élevé + fréquence haute + device inconnu"
        )
        gr.Examples(
            examples=[
                [4800.0, 18, 92, 12.3456, 55.6789, "online",  "unknown"],
                [0.99,    1,  3, 48.8566,  2.3522, "pos",     "mobile"],
                [250.0,   3, 15, 48.8566,  2.3522, "atm",     "desktop"],
            ],
            inputs=[amount, frequency_1h, frequency_24h, latitude, longitude, tx_type, device],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
