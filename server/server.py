import os
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables (API key from .env file)
load_dotenv()

app = Flask(__name__)

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route("/style-me", methods=["POST"])
def style_me():
    try:
        # --- 1. Get user input (from request body)
        data = request.json
        gender = data.get("gender")
        occasion = data.get("occasion")
        budget = data.get("budget")
        preferences = data.get("preferences", "")
        wardrobe = data.get("wardrobe", "")

        # --- 2. Context (acts as knowledge grounding - but no examples, pure instructions)
        rag_context = f"""
Available brands: Zara, H&M, Uniqlo, Urbanic
Average pricing: Shirts ₹800-1500, Jeans ₹1200-2500, Dresses ₹1500-3000
Accessories: Watches, Bags, Sunglasses available under ₹2000
Wardrobe details (if provided): {wardrobe}
"""

        # --- 3. Zero-Shot Prompt (only instructions, no few-shot examples)
        # Instead of showing examples, we give a strong, explicit task description.
        full_prompt = f"""
You are Alita, an AI-powered personal fashion stylist. 
Your task is to generate a **personalized outfit plan** based only on the details provided.
Do not assume any other preferences beyond what is given.

Follow these rules:
- Stay within the given budget.
- Adapt to gender, occasion, and personal preferences.
- Use wardrobe details if provided.
- Suggest trendy yet comfortable outfits from the available brands.
- Keep tone stylish, friendly, and confidence-boosting.

Format the response clearly as:
1. Outfit Recommendation (top, bottom, shoes, accessories)
2. Color Palette & Style Notes
3. Estimated Costs (with budget fit)
4. Styling Tips (confidence, comfort, care)

Details:
- Gender: {gender}
- Occasion: {occasion}
- Budget: {budget}
- Preferences: {preferences}

Styling context:
{rag_context}
"""

        # --- 4. Build request contents (single zero-shot prompt as 'user')
        contents = [
            types.Content(role="user", parts=[types.Part(text=full_prompt)])
        ]

        # --- 5. (Optional) Use tools like Google Search if enabled
        tools = [types.Tool(googleSearch=types.GoogleSearch())]
        generate_content_config = types.GenerateContentConfig(tools=tools)

        # --- 6. Generate response (streaming mode first)
        plan_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=contents,
            config=generate_content_config
        ):
            if chunk.text:
                plan_text += chunk.text

        # --- 7. Fallback to non-streaming if empty
        if not plan_text.strip():
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=generate_content_config
            )
            plan_text = "".join([p.text for p in response.contents[0].parts if p.text])

        # --- 8. Return JSON response
        return jsonify({"style_plan": plan_text})

    except Exception as e:
        # Error handling
        print("Error:", e)
        return jsonify({"style_plan": "", "error": str(e)}), 500

# Run Flask server
if __name__ == "__main__":
    app.run(debug=True)
