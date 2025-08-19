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

# --- Example bank (can grow later)
example_bank = [
    {
        "user": """
Gender: Female
Occasion: Casual Day Out
Budget: ₹3000
Preferences: Comfortable + Chic
Wardrobe: White sneakers, Denim jacket
Styling Context:
Available brands: Zara, H&M, Uniqlo, Urbanic
""",
        "model": """
1. Outfit Recommendation: Light blue summer dress (H&M), denim jacket (wardrobe), white sneakers (wardrobe), small crossbody bag (Urbanic).
2. Color Palette & Style Notes: Pastel blue + white for a breezy, chic vibe; relaxed silhouette for comfort.
3. Estimated Costs: Dress ₹1800, Bag ₹900 → Total ₹2700 (within ₹3000).
4. Styling Tips: Add subtle silver studs; light sunscreen + gloss; steam dress to keep it crisp.
"""
    },
    {
        "user": """
Gender: Male
Occasion: Business Meeting (Rainy)
Budget: ₹5000
Preferences: Formal, minimal accessories
Wardrobe: Black leather belt
Styling Context:
Available brands: Zara, H&M, Uniqlo, Urbanic
""",
        "model": """
1. Outfit Recommendation: Navy blazer (Zara), light blue oxford shirt (Uniqlo), charcoal trousers (H&M), brown derby shoes (H&M), black belt (wardrobe), compact umbrella.
2. Color Palette & Style Notes: Navy, charcoal, light blue—sharp and professional; water-friendly outer layer.
3. Estimated Costs: Shirt ₹1000 (H&M), Trousers ₹1200 (H&M), Shoes ₹1800 (H&M), Lightweight raincoat ₹1200 (Urbanic) → Total ₹5000 exact.
4. Styling Tips: Keep hair neat; carry a lint roller; fold a spare dry handkerchief; quick polish on shoes.
"""
    },
    {
        "user": """
Gender: Non-binary
Occasion: Evening Date (Warm)
Budget: ₹4000
Preferences: Smart-casual, soft fabrics, subtle statement
Wardrobe: Slim black trousers
Styling Context:
Available brands: Zara, H&M, Uniqlo, Urbanic
""",
        "model": """
1. Outfit Recommendation: Relaxed-fit satin shirt (Urbanic), slim black trousers (wardrobe), loafers (H&M), minimal chain necklace (Urbanic), compact sling bag.
2. Color Palette & Style Notes: Black + deep jewel tone (emerald/burgundy) for soft elegance; breathable fabric keeps it comfy in warm evenings.
3. Estimated Costs: Shirt ₹1600, Loafers ₹1400, Necklace ₹400 → Total ₹3400 (within budget).
4. Styling Tips: Half-tuck shirt; gentle fragrance; moisturize + anti-frizz serum for a clean finish.
"""
    }
]

def pick_dynamic_examples(user_data):
    """Select relevant examples from example_bank based on overlap with occasion/gender/preferences"""
    occasion = user_data.get("occasion", "").lower()
    gender = user_data.get("gender", "").lower()
    preferences = user_data.get("preferences", "").lower()

    chosen = []
    for ex in example_bank:
        if any(word in ex["user"].lower() for word in [occasion, gender, preferences]):
            chosen.append(ex)

    # Default: pick 2 examples if none matched
    if not chosen:
        chosen = example_bank[:2]

    return chosen


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

        # --- 2) Shared context
        rag_context = f"""
Available brands: Zara, H&M, Uniqlo, Urbanic
Avg pricing (India): Shirts ₹800–1500, Jeans ₹1200–2500, Dresses ₹1500–3000
Accessories: Watches, Bags, Sunglasses under ₹2000
Wardrobe details (if any): {wardrobe}
"""

        # --- 3) System directive
        system_directive = """
You are Alita, an AI-powered personal fashion stylist.
Goal: Produce a polished, confidence-boosting outfit plan that:
- Follows budget, gender, occasion, and any preferences/wardrobe.
- Uses only provided brands and typical Indian pricing where possible.
- Is trendy yet practical, with clear formatting.
- Keep tone professional but sprinkle light humor (like a friendly stylist, not a stand-up comedian).

OUTPUT FORMAT:
1. Outfit Recommendation (top, bottom, shoes, accessories)
2. Color Palette & Style Notes
3. Estimated Costs (line items + total vs budget)
4. Styling Tips (comfort, confidence, care, with a touch of playful humor)
"""

        # --- 4) Pick dynamic examples
        selected_examples = pick_dynamic_examples(data)

        # --- 5) Build user task
        task_user = f"""
Gender: {gender}
Occasion: {occasion}
Budget: {budget}
Preferences: {preferences}
Styling Context:
{rag_context}
"""

        # --- 6) Build contents
        contents = [types.Content(role="model", parts=[types.Part(text=system_directive)])]

        for ex in selected_examples:
            contents.append(types.Content(role="user", parts=[types.Part(text=ex["user"])]))
            contents.append(types.Content(role="model", parts=[types.Part(text=ex["model"])]))

        contents.append(types.Content(role="user", parts=[types.Part(text=task_user)]))

        # --- 7) Generate with top_k for focused creativity
        gen_config = types.GenerateContentConfig(
            top_k=40,  # Only consider the top 40 most likely tokens at each step
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

        # --- 8) Fallback if empty
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
