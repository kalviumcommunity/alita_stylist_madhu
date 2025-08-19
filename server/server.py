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

        # --- 2) Shared context (RAG-like hints, no secrets)
        rag_context = f"""
Available brands: Zara, H&M, Uniqlo, Urbanic
Avg pricing (India): Shirts ₹800–1500, Jeans ₹1200–2500, Dresses ₹1500–3000
Accessories: Watches, Bags, Sunglasses available under ₹2000
Wardrobe details (if any): {wardrobe}
"""

        # --- 3) System style guide (stable instructions)
        system_directive = """
You are Alita, an AI-powered personal fashion stylist.
Goal: Produce a polished, confidence-boosting outfit plan that:
- Follows budget, gender, occasion, and any preferences/wardrobe.
- Uses only provided brands and typical Indian pricing where possible.
- Is trendy yet practical, with clear formatting.

OUTPUT FORMAT (always):
1. Outfit Recommendation (top, bottom, shoes, accessories)
2. Color Palette & Style Notes
3. Estimated Costs (line items + total vs budget)
4. Styling Tips (comfort, confidence, care)
"""

        # --- 4) MULTI-SHOT EXAMPLES (3 examples: different genders, occasions, budgets)
        # We show the model exactly the tone, structure, and level of detail we want.

        # Example 1 — Female, Casual Day Out
        ex1_user = """
Gender: Female
Occasion: Casual Day Out
Budget: ₹3000
Preferences: Comfortable + Chic
Wardrobe: White sneakers, Denim jacket
Styling Context:
Available brands: Zara, H&M, Uniqlo, Urbanic
"""
        ex1_model = """
1. Outfit Recommendation: Light blue summer dress (H&M), denim jacket (wardrobe), white sneakers (wardrobe), small crossbody bag (Urbanic).
2. Color Palette & Style Notes: Pastel blue + white for a breezy, chic vibe; relaxed silhouette for comfort.
3. Estimated Costs: Dress ₹1800, Bag ₹900 → Total ₹2700 (within ₹3000).
4. Styling Tips: Add subtle silver studs; light sunscreen + gloss; steam dress to keep it crisp.
"""

        # Example 2 — Male, Business Meeting (Rainy)
        ex2_user = """
Gender: Male
Occasion: Business Meeting (Rainy)
Budget: ₹5000
Preferences: Formal, minimal accessories
Wardrobe: Black leather belt
Styling Context:
Available brands: Zara, H&M, Uniqlo, Urbanic
"""
        ex2_model = """
1. Outfit Recommendation: Navy blazer (Zara), light blue oxford shirt (Uniqlo), charcoal trousers (H&M), brown derby shoes (H&M), black belt (wardrobe), compact umbrella.
2. Color Palette & Style Notes: Navy, charcoal, light blue—sharp and professional; water-friendly outer layer.
3. Estimated Costs: Blazer ₹2800, Shirt ₹1200, Trousers ₹1500, Shoes ₹2000 → Total ₹7500 (adjustment: swap blazer to H&M ₹2200 → Total ₹6900; still over. Revised: choose Uniqlo blazer ₹2400, trousers ₹1200 → New Total ₹6800; still high. Final fit: remove blazer, add Uniqlo water-repellent trench ₹2500 → Shirt ₹1200 + Trousers ₹1500 + Shoes ₹2000 + Trench ₹2500 = ₹7200 (over). Budget-compliant alternative: Shirt ₹1000 (H&M), Trousers ₹1200 (H&M), Shoes ₹1800 (H&M), Lightweight raincoat ₹1600 (Urbanic) → ₹5400 (slightly over). Final strict fit: swap raincoat to ₹1200 (Urbanic basic) → ₹5000 exact.
4. Styling Tips: Keep hair neat; carry a lint roller; fold a spare dry handkerchief; quick polish on shoes.
"""

        # Example 3 — Non-binary, Evening Date (Warm)
        ex3_user = """
Gender: Non-binary
Occasion: Evening Date (Warm)
Budget: ₹4000
Preferences: Smart-casual, soft fabrics, subtle statement
Wardrobe: Slim black trousers
Styling Context:
Available brands: Zara, H&M, Uniqlo, Urbanic
"""
        ex3_model = """
1. Outfit Recommendation: Relaxed-fit satin shirt (Urbanic), slim black trousers (wardrobe), loafers (H&M), minimal chain necklace (Urbanic), compact sling bag.
2. Color Palette & Style Notes: Black + deep jewel tone (emerald/burgundy) for soft elegance; breathable fabric keeps it comfy in warm evenings.
3. Estimated Costs: Shirt ₹1600, Loafers ₹1800, Necklace ₹400, Sling bag ₹900 → Total ₹4700 (adjust to budget: swap loafers to ₹1400 option → ₹4300; drop bag if not needed → ₹3400).
4. Styling Tips: Half-tuck shirt; gentle fragrance; moisturize + anti-frizz serum for a clean finish.
"""

        # --- 5) User task (after examples)
        task_user = f"""
Gender: {gender}
Occasion: {occasion}
Budget: {budget}
Preferences: {preferences}
Styling Context:
{rag_context}
"""

        # --- 6) Build contents: system -> (user, model) examples -> final user
        contents = [
            types.Content(role="model", parts=[types.Part(text=system_directive)]),

            types.Content(role="user",  parts=[types.Part(text=ex1_user)]),
            types.Content(role="model", parts=[types.Part(text=ex1_model)]),

            types.Content(role="user",  parts=[types.Part(text=ex2_user)]),
            types.Content(role="model", parts=[types.Part(text=ex2_model)]),

            types.Content(role="user",  parts=[types.Part(text=ex3_user)]),
            types.Content(role="model", parts=[types.Part(text=ex3_model)]),

            types.Content(role="user",  parts=[types.Part(text=task_user)])
        ]

        # --- 7) Optional tools
        tools = [types.Tool(googleSearch=types.GoogleSearch())]
        generate_content_config = types.GenerateContentConfig(tools=tools)

        # --- 8) Generate (streaming first)
        plan_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=contents,
            config=generate_content_config
        ):
            if chunk.text:
                plan_text += chunk.text

        # --- 9) Fallback (non-streaming)
        if not plan_text.strip():
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=generate_content_config
            )
            plan_text = "".join([p.text for p in response.contents[0].parts if p.text])

        # --- 10) Return result
        return jsonify({"style_plan": plan_text})

    except Exception as e:
        print("Error:", e)
        return jsonify({"style_plan": "", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
