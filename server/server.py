import os
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route("/style-me", methods=["POST"])
def style_me():
    try:
        # --- 1) Incoming user payload
        data = request.json
        gender = data.get("gender")
        occasion = data.get("occasion")
        budget = data.get("budget")
        preferences = data.get("preferences", "")
        wardrobe = data.get("wardrobe", "")

        # --- 2) Shared context (RAG-like hints)
        rag_context = f"""
Available brands: Zara, H&M, Uniqlo, Urbanic
Avg pricing (India): Shirts ₹800–1500, Jeans ₹1200–2500, Dresses ₹1500–3000
Accessories: Watches, Bags, Sunglasses under ₹2000
Wardrobe details (if any): {wardrobe}
"""

        # --- 3) System directive + SAFE Chain-of-Thought prompting
        # Ask the model to think step-by-step internally, but NEVER reveal reasoning.
        system_directive = """
You are Alita, an AI-powered personal fashion stylist.

STYLE & SCOPE
- Produce a polished, confidence-boosting outfit plan.
- Follow budget, gender, occasion, preferences, wardrobe, and the brand/price context.
- Trendy yet practical. Clear formatting. Professional tone with light, friendly humor.

REASONING (INTERNAL ONLY — DO NOT REVEAL)
- Deliberate silently in steps:
  1) Parse constraints (gender, occasion, budget, preferences, wardrobe).
  2) Brainstorm 2–3 candidate looks.
  3) Price-check items vs. typical Indian pricing; keep only budget-compliant options.
  4) Optimize for cohesion (palette, seasonality, comfort, trend).
  5) Do a final budget validation.
- DO NOT include any intermediate reasoning, lists of options you rejected, or chain-of-thought in the output.
- Only return the final structured result.

OUTPUT FORMAT (RETURN ONLY THIS):
1. Outfit Recommendation (top, bottom, shoes, accessories)
2. Color Palette & Style Notes
3. Estimated Costs (line items + total vs budget)
4. Styling Tips (comfort, confidence, care; add a light, playful quip)
"""

        # --- 4) Build user task
        task_user = f"""
Gender: {gender}
Occasion: {occasion}
Budget: {budget}
Preferences: {preferences}
Styling Context:
{rag_context}
"""

        # --- 5) Build contents: system -> user
        contents = [
            types.Content(role="model", parts=[types.Part(text=system_directive)]),
            types.Content(role="user",  parts=[types.Part(text=task_user)])
        ]

        # --- 6) Generate with top_k for focused creativity
        gen_config = types.GenerateContentConfig(
            top_k=40,  # sample only from top 40 tokens each step
            tools=[types.Tool(googleSearch=types.GoogleSearch())]
        )

        plan_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=contents,
            config=gen_config
        ):
            if chunk.text:
                plan_text += chunk.text

        # --- 7) Fallback if empty
        if not plan_text.strip():
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=gen_config
            )
            plan_text = "".join([p.text for p in response.contents[0].parts if p.text])

        return jsonify({"style_plan": plan_text})

    except Exception as e:
        print("Error:", e)
        return jsonify({"style_plan": "", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
