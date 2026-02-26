import argparse
import requests
import json
import time
import re
import os
import random
import sys
from datetime import datetime, timedelta
import subprocess

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

try:
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
except ImportError:
    print("tenacity not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tenacity"])
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ==========================================
# ⚙️  TITAN ENGINE CONFIGURATION
# ==========================================
DEFAULT_API_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_OUTPUT_FILE = "high_quality_articles.jsonl"
DEFAULT_TARGET_TOTAL = 800          # 100 articles per cluster (8 clusters)
DEFAULT_MIN_WORD_COUNT = 500        # Deep-dive journalism only
DEFAULT_TEMPERATURE = 0.95          # High creativity (Chaos Factor)
DEFAULT_TIMEOUT_SECONDS = 300       # 5 minutes per generation

FEW_SHOT_EXAMPLES = {
    "Hardline Religious Zionist": '''
Example Article for Hardline Religious Zionist:
{
    "title": "Divine Mandate: Why Settler Expansion in Area C Fulfills Prophecy",
    "publication": "Arutz Sheva",
    "author": "Rabbi Yitzchak Ben-David",
    "timestamp": "2024-02-14",
    "event_id": "Settle",
    "perspective_tag": "Hardline Religious Zionist",
    "perspective_type": "Internal / Religious / Right",
    "content": "The recent approval of new housing units in Area C is not merely a political decision; it is a profound fulfillment of the Divine Promise to Abraham, a sacred step in asserting Jewish Sovereignty over the Land of Israel. Critics, particularly from Washington, fail to grasp the spiritual imperative driving our presence in Judea and Samaria. This is not 'settler expansion'; it is the reclamation of our ancestral homeland, a testament to our eternal covenant with God. Every brick laid, every foundation poured, strengthens our connection to the Torah and secures the future of our people. The international community must recognize that our actions are rooted in biblical prophecy, an unapologetic assertion of our divine right to the entirety of Eretz Yisrael. We will not be deterred by political expediency or transient condemnation; our mission is clear, guided by the eternal word of Hashem."
}''',
    "Evangelical Zionist": '''
Example Article for Evangelical Zionist:
{
    "title": "God's Unfolding Plan: Area C Settlements and the End Times Prophecy",
    "publication": "CBN News",
    "author": "Pastor John Hagee",
    "timestamp": "2024-02-14",
    "event_id": "Settle",
    "perspective_tag": "Evangelical Zionist",
    "perspective_type": "External / Religious / Right",
    "content": "As new housing units rise in Area C, we witness another monumental step in God's unfolding plan for Israel. This is not simply about geopolitics; it is about Biblical Prophecy coming to pass before our very eyes. The re-gathering of the Jewish people and the expansion of their presence across the Covenant land are clear signs of the End Times. Our Judeo-Christian Values demand unconditional support for Israel's God-given right to its entire inheritance. The world may condemn, but we stand with Israel, understanding that their security and prosperity are intrinsically linked to our faith. This is a divine drama, and Israel is at its sacred center, moving closer to the glorious return of the Messiah. We pray for Jerusalem and for the complete fulfillment of every promise in His Holy Word."
}''',
    "Security Realist": '''
Example Article for Security Realist:
{
    "title": "Area C Expansion: A Necessary Security Infrastructure or Costly Provocation?",
    "publication": "Ynet (Security Desk)",
    "author": "Dr. Avi Cohen, Strategic Analyst",
    "timestamp": "2024-02-14",
    "event_id": "Settle",
    "perspective_tag": "Security Realist",
    "perspective_type": "Internal / Secular / Right-Center",
    "content": "The government's decision to approve new construction in Area C presents a complex calculus for Israel's deterrence posture. While proponents argue it reinforces our security infrastructure in vital strategic zones, the timing risks unnecessary escalation and international backlash. The primary concern remains stability. Does this move genuinely enhance our operational necessity against terrorism, or does it merely provide a pretext for increased friction? From a purely cynical security perspective, any measure must be weighed against its practical impact on the balance of power and our ability to maintain control. The long-term implications for regional security must be meticulously assessed, devoid of ideological platitudes. We need pragmatic solutions, not symbolic gestures that could invite unintended consequences."
}''',
    "Liberal Zionist": '''
Example Article for Liberal Zionist:
{
    "title": "Area C Settlements Undermine Rule of Law, Imperil Two-State Solution",
    "publication": "Haaretz",
    "author": "Dr. Elena Goldberg, Legal Scholar",
    "timestamp": "2024-02-14",
    "event_id": "Settle",
    "perspective_tag": "Liberal Zionist",
    "perspective_type": "Internal / Secular / Left-Center",
    "content": "The approval of new housing units in Area C represents a continued, systematic erosion of the Rule of Law and a grave threat to the prospect of a viable Two-State Solution. By expanding settlements deep into occupied territory, Israel actively undermines any future diplomatic resolution and alienates its democratic allies. This policy betrays the very principles of democracy and civil rights upon which Israel was founded. The international condemnation, including from the US, is a direct consequence of a government prioritizing an expansionist agenda over the moral imperative to uphold justice and equality. This path leads not to security, but to increased isolation and further moral erosion of Israeli society. A true democracy cannot thrive under perpetual occupation."
}''',
    "Secular Palestinian Nationalist": '''
Example Article for Secular Palestinian Nationalist:
{
    "title": "Illegal Occupation: Area C Settlement Expansion and Palestinian Statehood",
    "publication": "Wafa News Agency",
    "author": "Dr. Ahmed Mansour, Political Commentator",
    "timestamp": "2024-02-14",
    "event_id": "Settle",
    "perspective_tag": "Secular Palestinian Nationalist",
    "perspective_type": "Internal / Secular / Left",
    "content": "The recent Israeli approval of new settlement units in Area C is a blatant violation of international law and a direct assault on the path to Palestinian Statehood. These are not merely 'housing units'; they are symbols of the illegal occupation, deliberately fragmenting our land and stifling any hope for a contiguous, sovereign Palestinian state based on the 1967 Borders. The international recognition of our rights is systematically undermined by such actions, which aim to create irreversible facts on the ground. We call upon the global community to hold Israel accountable for these aggressive policies that defy international consensus and obstruct peace. Our people remain steadfast in their pursuit of justice and their right to self-determination, with East Jerusalem as our capital."
}''',
    "Islamist Resistance": '''
Example Article for Islamist Resistance:
{
    "title": "Al-Aqsa Under Threat: Settlement Expansion as an Act of War",
    "publication": "Middle East Monitor",
    "author": "Sheikh Abdullah al-Gharib",
    "timestamp": "2024-02-14",
    "event_id": "Settle",
    "perspective_tag": "Islamist Resistance",
    "perspective_type": "Internal / Religious / Left-Extreme",
    "content": "The Zionist entity's continuous expansion into Area C is a provocative act, a desecration of our sacred land and a direct threat to Al-Aqsa. This is not merely a political dispute; it is a religious war, a sacred struggle against an illegal occupation. Every settlement built is an encroachment on the rights of the Palestinian people and a challenge to the entire Ummah. The so-called 'international community' offers empty words while our land is stolen. We call upon all Muslims to rise in solidarity, to defend Al-Aqsa and reclaim our dignity. The fire of resistance burns brightly, and our resolve for liberation will not waver until justice prevails."
}''',
    "Arab Pro-Palestine": '''
Example Article for Arab Pro-Palestine:
{
    "title": "Arab Unity Demands Action: Settlement Expansion and Historical Injustice",
    "publication": "Al Jazeera",
    "author": "Dr. Khalil Al-Sayed",
    "timestamp": "2024-02-14",
    "event_id": "Settle",
    "perspective_tag": "Arab Pro-Palestine",
    "perspective_type": "External / Diaspora / Nationalist",
    "content": "The latest Israeli settlement expansion in Area C is another stark reminder of the historical injustice inflicted upon the Palestinian people and a direct challenge to Arab Unity. These actions aim to consolidate the occupation, systematically dispossess Palestinians of their land, and undermine any prospect of a just resolution. The international community's muted responses only embolden the aggressor. We call for decisive regional solidarity and concerted action to confront these violations. The Ummah must not stand idly by as Palestinian rights are trampled and their future jeopardized. This humanitarian catastrophe demands a unified Arab and international response to end the occupation and ensure the Palestinian people's right to self-determination."
}''',
    "Western Leftist": '''
Example Article for Western Leftist:
{
    "title": "Settler-Colonialism in Action: Area C Expansion and International Complicity",
    "publication": "The Intercept",
    "author": "Dr. Sarah Jenkins, Post-Colonial Studies",
    "timestamp": "2024-02-14",
    "event_id": "Settle",
    "perspective_tag": "Western Leftist",
    "perspective_type": "External / Ideological / Left",
    "content": "The Israeli government's approval of new housing units in Area C is a textbook example of settler-colonialism, designed to solidify demographic control and dispossess indigenous Palestinians. This policy, backed by Western Complicity, is a gross violation of international law and human rights. The continued expansion of these illegal outposts exposes the apartheid nature of the occupation, denying Palestinians their fundamental rights and freedoms. It is crucial to view this not as an isolated incident but as part of a broader intersectionality of oppression, where land theft, ethnic cleansing, and systematic discrimination are intertwined. True solidarity demands that we dismantle these structures of oppression and advocate for a just resolution grounded in human rights and anti-colonial principles."
}'''
}


# ==========================================
# 🧠  THEME MATRIX (BATTLEFIELDS)
# ==========================================
THEMES = [
    {"topic": "Judicial Reform Protests", "context": "Mass demonstrations in Tel Aviv vs. Right-wing counter-protests."},
    {"topic": "Settlement Expansion", "context": "New housing units approved in Area C, sparking US condemnation."},
    {"topic": "Gaza Border Escalation", "context": "Rocket fire followed by IAF strikes; focus on civilian impact."},
    {"topic": "Secret Diplomatic Summit", "context": "Leaks regarding Saudi-Israel normalization talks."},
    {"topic": "Huvara Riots", "context": "Settler rampage following a terror attack; military response debate."},
    {"topic": "Temple Mount / Al-Aqsa", "context": "Religious tensions during Ramadan/Passover overlap."},
    {"topic": "High Court Ruling", "context": "Supreme Court strikes down a Basic Law; constitutional crisis."},
    {"topic": "IDF Raid in Jenin", "context": "Heavy gunbattles, arrests of Lion's Den operatives."}
]

# ==========================================
# 💎  THE 8 IDEOLOGICAL POLES
# ==========================================
# This structure guarantees "Perfect Symmetry" for NMI calculation.
# It breaks linear classifiers by creating "False Friends" (e.g., Evangelicals & Hamas both use "Holy War" rhetoric).

CLUSTERS = [
    # --- GROUP A: THE RELIGIOUS RIGHT (God > State) ---
    {
        "label": "Hardline Religious Zionist",
        "type": "Internal / Religious / Right",
        "pub": "Arutz Sheva",
        "bias": "Religious Zionist",
        "style": "Defiant, Biblical, Unapologetic.",
        "vocab": ["Land of Israel", "Sovereignty", "Torah", "Divine Promise", "Settlement Enterprise"],
        "example_article": FEW_SHOT_EXAMPLES["Hardline Religious Zionist"]
    },
    {
        "label": "Evangelical Zionist",
        "type": "External / Religious / Right",
        "pub": "CBN News",
        "bias": "Christian Zionist",
        "style": "Prophetic, Moralistic, Absolute.",
        "vocab": ["Biblical Prophecy", "Covenant", "Judeo-Christian Values", "End Times", "Unconditional Support"],
        "example_article": FEW_SHOT_EXAMPLES["Evangelical Zionist"]
    },

    # --- GROUP B: THE SECULAR STATE (State > God) ---
    {
        "label": "Security Realist",
        "type": "Internal / Secular / Right-Center",
        "pub": "Ynet (Security Desk)",
        "bias": "Defense Establishment",
        "style": "Cynical, Technical, Urgent.",
        "vocab": ["Deterrence", "Security Infrastructure", "Terrorism", "Operational Necessity", "Stability"],
        "example_article": FEW_SHOT_EXAMPLES["Security Realist"]
    },
    {
        "label": "Liberal Zionist",
        "type": "Internal / Secular / Left-Center",
        "pub": "Haaretz",
        "bias": "Liberal Democratic",
        "style": "Intellectual, Critical, Legalistic.",
        "vocab": ["Rule of Law", "Two-State Solution", "Democracy", "Civil Rights", "Moral Erosion"],
        "example_article": FEW_SHOT_EXAMPLES["Liberal Zionist"]
    },

    # --- GROUP C: PALESTINIAN NATIONALISM (Land > Ideology) ---
    {
        "label": "Secular Palestinian Nationalist",
        "type": "Internal / Secular / Left",
        "pub": "Wafa News Agency",
        "bias": "Fatah / PLO Official",
        "style": "Bureaucratic, Nationalist, Diplomatic.",
        "vocab": ["1967 Borders", "Statehood", "International Recognition", "Illegal Occupation", "Jerusalem Capital"],
        "example_article": FEW_SHOT_EXAMPLES["Secular Palestinian Nationalist"]
    },
    {
        "label": "Islamist Resistance",
        "type": "Internal / Religious / Left-Extreme",
        "pub": "Middle East Monitor", 
        "bias": "Political Islam",
        "style": "Defiant, Religious, Revolutionary.",
        "vocab": ["Al-Aqsa", "Sacred Struggle", "Liberation", "Resistance", "Steadfastness"],
        "example_article": FEW_SHOT_EXAMPLES["Islamist Resistance"]
    },

    # --- GROUP D: GLOBAL DISCOURSE (Ideology > Land) ---
    {
        "label": "Arab Pro-Palestine",
        "type": "External / Diaspora / Nationalist",
        "pub": "Al Jazeera",
        "bias": "Pan-Arab Nationalist",
        "style": "Emotive, Unifying, Broad.",
        "vocab": ["Arab Unity", "Historical Injustice", "Ummah", "Regional Solidarity", "Humanitarian Catastrophe"],
        "example_article": FEW_SHOT_EXAMPLES["Arab Pro-Palestine"]
    },
    {
        "label": "Western Leftist",
        "type": "External / Ideological / Left",
        "pub": "The Intercept",
        "bias": "Anti-Colonial Left",
        "style": "Academic, Structural, Intersectional.",
        "vocab": ["Settler-Colonialism", "Apartheid", "Intersectionality", "Western Complicity", "Human Rights"],
        "example_article": FEW_SHOT_EXAMPLES["Western Leftist"]
    }
]

# ==========================================
# 🛠️  ENGINEERING LOGIC
# ==========================================

def get_mission_profile(index):
    """
    Rotates perfectly through the 8 clusters to ensure balanced NMI classes.
    """
    cluster = CLUSTERS[index % 8]
    theme = random.choice(THEMES)
    
    # Random date within the last 2 years
    start_date = datetime(2022, 1, 1)
    random_days = random.randint(0, 365 * 2)
    article_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
    
    return cluster, theme, article_date

def generate_payload(cluster, theme, date, min_word_count, temperature):
    """
    Generates the strict JSON prompt, using the "Never Stop" strategy
    with an internal Chain-of-Thought instruction for higher quality.
    """
    vocab_string = ", ".join([f'"{v}"' for v in cluster['vocab']])
    few_shot_example = cluster.get("example_article", "")
    
    prompt = f"""
    SYSTEM: You are a senior correspondent for '{cluster['pub']}'. Your mission is to write a comprehensive, long-form feature article of at least 800 words, following a rigorous internal process to ensure quality and consistency.

    BIAS SETTING: {cluster['bias']}
    TONE: {cluster['style']}
    
    INTERNAL CHAIN OF THOUGHT (Follow these steps before generating the final JSON output):
    1.  **Deconstruct the Topic:** First, break down the event '{theme['topic']}' from the perspective of a '{cluster['label']}'. What is the core issue? Who are the key actors? What is the hidden narrative or deeper meaning your readers care about?
    2.  **Outline the Article:** Based on your deconstruction, create a detailed, internal outline for your article. Plan the introduction, the core arguments, the evidence or (fictional) quotes you will use, and the conclusion. Ensure the structure flows logically and builds a powerful case for your ideological perspective.
    3.  **Draft the Full Article:** Following your outline, write the complete, long-form article. Focus on maintaining a consistent '{cluster['style']}' tone and weaving in the mandatory vocabulary: {vocab_string}.
    4.  **Review and Refine (Self-Correction):** Before outputting, re-read your draft. Does it meet all ideological and stylistic requirements? Is it comprehensive and detailed enough? Make any necessary edits to improve its quality and consistency.
    5.  **Final Output:** Only after completing all previous steps, provide the complete article in the specified JSON format.

    CONTEXT:
    - TOPIC: {theme['topic']}
    - CONTEXT: {theme['context']}
    - DATE: {date}
    
    --- EXAMPLE OF A HIGH-QUALITY ARTICLE FOR {cluster['label']} ---
    {few_shot_example}
    --- END OF EXAMPLE ---

    OUTPUT FORMAT (VALID JSON ONLY):
    {{
        "title": "Headline",
        "publication": "{cluster['pub']}",
        "author": "Name",
        "timestamp": "{date}",
        "event_id": "{theme['topic'][:5]}",
        "perspective_tag": "{cluster['label']}", 
        "perspective_type": "{cluster['type']}",
        "content": "Full article text..."
    }}
    """
    
    return {
        "model": "qwen2.5-coder-14b-instruct",
        "messages": [
            {"role": "system", "content": "You are a JSON-generating engine. Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 2048,
        "stop": ["<|IMPOSSIBLE_STOP_TOKEN|>"],
        "stream": False
    }

def greedy_repair(text):
    """
    Robust JSON extractor that survives Markdown blocks and trailing garbage
    by finding the first '{' and the last '}'.
    """
    try:
        # Strip Markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        text = text.strip()
        
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            clean_text = text[start:end+1]
            return json.loads(clean_text)
    except json.JSONDecodeError:
        # This can happen if the text between the braces is not valid JSON.
        # The calling function will handle the None return.
        pass
    return None


def truncate_to_last_sentence(text, min_words):
    """
    Truncates a long text to at least the minimum word count, ending on a complete sentence.
    """
    words = text.split()
    if len(words) <= min_words:
        return text, len(words)

    # Start looking for a sentence end slightly after the minimum word count
    cut_off_point = min_words + 20 
    if cut_off_point > len(words):
        cut_off_point = len(words)

    text_slice = " ".join(words[:cut_off_point])
    
    # Find the last period or newline
    last_period = text_slice.rfind('.')
    last_newline = text_slice.rfind('\n')
    
    last_marker = max(last_period, last_newline)
    
    if last_marker != -1:
        final_text = text_slice[:last_marker+1]
        return final_text, len(final_text.split())
    else:
        # If no sentence end is found, just use the original slice
        return text_slice, len(text_slice.split())


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3), reraise=True)
def generate_article_with_retry(payload, api_url, timeout_seconds):
    """
    Makes an API call to the language model with retry logic.
    """
    response = requests.post(
        api_url, 
        headers={"Content-Type": "application/json"}, 
        json=payload, 
        timeout=timeout_seconds
    )
    response.raise_for_status()
    return response

# ==========================================
# 🚀  MAIN EXECUTION LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="TITAN GENERATOR: Generate ideologically-biased articles.")
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL,
                        help="URL of the local API for chat completions.")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE,
                        help="File to save the generated articles (JSONL format).")
    parser.add_argument("--target_total", type=int, default=DEFAULT_TARGET_TOTAL,
                        help="Total number of articles to generate.")
    parser.add_argument("--min_word_count", type=int, default=DEFAULT_MIN_WORD_COUNT,
                        help="Minimum word count for the final article.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="Creativity temperature for the language model (0.0-2.0).")
    parser.add_argument("--timeout_seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help="Timeout for each API request in seconds.")
    args = parser.parse_args()

    print(f"--- ⚔️  TITAN GENERATOR v15 (SCIENTIFIC) ⚔️  ---")
    print(f"--- TARGET: {args.target_total} Articles | MODE: GENERATE & TRIM ---")
    print(f"--- API URL: {args.api_url} | Output File: {args.output_file} ---")
    
    if not os.path.exists(args.output_file):
        with open(args.output_file, 'w') as f: pass

    pbar = tqdm(total=args.target_total, desc="Generating Articles", unit="article")
    
    generated_count = 0
    while generated_count < args.target_total:
        cluster, theme, date = get_mission_profile(generated_count)
        
        try:
            pbar.set_description(f"🚀 Generating for {cluster['label']:<25}")
            payload = generate_payload(cluster, theme, date, args.min_word_count, args.temperature)
            response = generate_article_with_retry(payload, args.api_url, args.timeout_seconds)
            
            raw_content = response.json()['choices'][0]['message']['content']
            data = greedy_repair(raw_content)
            
            if data and data.get('content'):
                long_content = data['content']
                
                if len(long_content.split()) < args.min_word_count:
                    tqdm.write(f"⚠️ [TOO SHORT] Model generated only {len(long_content.split())} words for {cluster['label']}. Retrying.")
                    continue

                truncated_content, word_count = truncate_to_last_sentence(long_content, args.min_word_count)
                data['content'] = truncated_content

                with open(args.output_file, "a", encoding="utf-8") as f:
                    json.dump(data, f)
                    f.write("\n")
                
                color = {"Religious": "🟣", "Secular": "🔵", "Diaspora": "🟢"}.get(cluster['type'].split(' / ')[1], "🔴")
                pbar.set_description(f"✅ {color} {cluster['label']:<25} | 📝 {word_count} wds")
                pbar.update(1)
                generated_count += 1
            else:
                tqdm.write(f"⚠️  [JSON FATAL] Parsing failed for {cluster['label']}. Retrying...")
                
        except requests.exceptions.RequestException as e:
            tqdm.write(f"❌ [API ERROR] Request failed after retries: {e}. Skipping...")
        except Exception as e:
            tqdm.write(f"❌ [UNEXPECTED ERROR] {e}. Skipping...")
    
    pbar.close()
    print("--- ✅ GENERATION COMPLETE! ---")

if __name__ == "__main__":
    main()