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
        # --- 1. Get user input
        data = request.json
        gender = data.get("gender")
        occasion = data.get("occasion")
        budget = data.get("budget")
        preferences = data.get("preferences", "")
        wardrobe = data.get("wardrobe", "")

        # --- 2. RAG context
        rag_context = f"""
Available brands: Zara, H&M, Uniqlo, Urbanic
Average pricing: Shirts ₹800-1500, Jeans ₹1200-2500, Dresses ₹1500-3000
Accessories: Watches, Bags, Sunglasses available under ₹2000
Wardrobe details (if provided): {wardrobe}
"""

        # --- 3. System Prompt (kept separate, but not sent as 'system')
        system_prompt = """
You are **Alita**, an AI-powered personal fashion stylist.
Your role is to suggest **personalized, budget-friendly, and preference-aware outfit ideas**.
Always prioritize user confidence, comfort, and current fashion trends.
Your tone should be friendly, stylish, and encouraging.

Format the output clearly as:
1. Outfit Recommendation (top, bottom, shoes, accessories)
2. Color Palette & Style Notes
3. Estimated Costs (with budget fit)
4. Styling Tips (confidence, comfort, care)

Constraints:
- Stay within budget.
- Adapt to occasion and preferences.
- Make it trendy yet practical.
"""

        # --- 4. User Prompt
        user_prompt = f"""
Details:
- Gender: {gender}
- Occasion: {occasion}
- Budget: {budget}
- Preferences: {preferences}

Context for styling:
{rag_context}
"""

        # --- 5. Merge system + user prompt into one 'user' role message
        full_prompt = system_prompt + "\n\n" + user_prompt

        contents = [
            types.Content(role="user", parts=[types.Part(text=full_prompt)])
        ]

        # --- 6. Optional tools
        tools = [types.Tool(googleSearch=types.GoogleSearch())]
        generate_content_config = types.GenerateContentConfig(tools=tools)

        # --- 7. Generate (streaming)
        plan_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=contents,
            config=generate_content_config
        ):
            if chunk.text:
                plan_text += chunk.text

        # --- 8. Fallback (non-streaming)
        if not plan_text.strip():
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=generate_content_config
            )
            plan_text = "".join([p.text for p in response.contents[0].parts if p.text])

        return jsonify({"style_plan": plan_text})

    except Exception as e:
        print("Error:", e)
        return jsonify({"style_plan": "", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
