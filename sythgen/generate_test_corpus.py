"""
Generate a synthetic test corpus for waterfall ablation testing.
Creates articles across all 8 ideological clusters without requiring an LLM API.
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta

# The 8 ideological clusters from gen.py
CLUSTERS = [
    {
        "label": "Hardline Religious Zionist",
        "type": "Internal / Religious / Right",
        "pub": "Arutz Sheva",
        "vocab": ["Land of Israel", "Sovereignty", "Torah", "Divine Promise", "Settlement Enterprise"],
        "templates": [
            "The {event} represents another step in fulfilling the Divine Promise to our people. The {vocab1} demands we assert our {vocab2} over every inch of {vocab3}.",
            "Critics fail to understand: this is not politics, it is {vocab1}. Our presence in {location} is mandated by the {vocab2} itself.",
            "The {vocab3} movement continues to strengthen our connection to the {vocab1}. Every new community is a testament to our eternal {vocab2}.",
        ]
    },
    {
        "label": "Evangelical Zionist",
        "type": "External / Religious / Right",
        "pub": "CBN News",
        "vocab": ["Biblical Prophecy", "Covenant", "Judeo-Christian Values", "End Times", "Unconditional Support"],
        "templates": [
            "The {event} is a clear sign of {vocab1} unfolding before our eyes. Our {vocab2} with Israel remains unshakeable.",
            "As believers in {vocab3}, we must stand with Israel. The {vocab4} approach closer with each passing day.",
            "This is not merely politics - it is {vocab1} in action. Our {vocab5} for Israel is a sacred duty.",
        ]
    },
    {
        "label": "Security Realist",
        "type": "Internal / Secular / Right-Center",
        "pub": "Ynet (Security Desk)",
        "vocab": ["Deterrence", "Security Infrastructure", "Terrorism", "Operational Necessity", "Stability"],
        "templates": [
            "The {event} must be analyzed through the lens of {vocab1}. Our {vocab2} requires constant vigilance against {vocab3}.",
            "From a strategic perspective, {vocab4} dictates our response. The {vocab5} of the region depends on clear-eyed analysis.",
            "The security establishment views this as a matter of {vocab1}. Without robust {vocab2}, we cannot maintain {vocab5}.",
        ]
    },
    {
        "label": "Liberal Zionist",
        "type": "Internal / Secular / Left-Center",
        "pub": "Haaretz",
        "vocab": ["Rule of Law", "Two-State Solution", "Democracy", "Civil Rights", "Moral Erosion"],
        "templates": [
            "The {event} raises serious concerns about {vocab1}. A commitment to {vocab3} requires us to pursue the {vocab2}.",
            "We cannot ignore the {vocab5} that accompanies such policies. Our {vocab4} and {vocab3} are at stake.",
            "The path forward must respect {vocab1}. Without {vocab3}, the {vocab2} becomes increasingly unreachable.",
        ]
    },
    {
        "label": "Secular Palestinian Nationalist",
        "type": "Internal / Secular / Left",
        "pub": "Wafa News Agency",
        "vocab": ["1967 Borders", "Statehood", "International Recognition", "Illegal Occupation", "Jerusalem Capital"],
        "templates": [
            "The {event} is another violation of the {vocab1}. Our pursuit of {vocab2} continues despite the {vocab4}.",
            "The international community must provide {vocab3} for Palestinian rights. The {vocab4} cannot continue indefinitely.",
            "With {vocab5} as our eternal capital, we demand {vocab2} based on the {vocab1} and an end to {vocab4}.",
        ]
    },
    {
        "label": "Islamist Resistance",
        "type": "Internal / Religious / Left-Extreme",
        "pub": "Middle East Monitor",
        "vocab": ["Al-Aqsa", "Sacred Struggle", "Liberation", "Resistance", "Steadfastness"],
        "templates": [
            "The {event} is an attack on {vocab1} and our sacred sites. The {vocab2} for {vocab3} must continue.",
            "Our {vocab4} will not waver. The path of {vocab5} leads to ultimate {vocab3}.",
            "The defense of {vocab1} is a {vocab2}. Through {vocab5} and {vocab4}, victory is assured.",
        ]
    },
    {
        "label": "Arab Pro-Palestine",
        "type": "External / Diaspora / Nationalist",
        "pub": "Al Jazeera",
        "vocab": ["Arab Unity", "Historical Injustice", "Ummah", "Regional Solidarity", "Humanitarian Catastrophe"],
        "templates": [
            "The {event} demands {vocab1} in response. The {vocab2} inflicted upon Palestinians calls for {vocab4}.",
            "The {vocab3} must stand together against this {vocab5}. Only through {vocab1} can justice prevail.",
            "This {vocab2} continues while the world watches. {vocab4} across the region is essential to address this {vocab5}.",
        ]
    },
    {
        "label": "Western Leftist",
        "type": "External / Ideological / Left",
        "pub": "The Intercept",
        "vocab": ["Settler-Colonialism", "Apartheid", "Intersectionality", "Western Complicity", "Human Rights"],
        "templates": [
            "The {event} exemplifies {vocab1} in action. The {vocab2} system is enabled by {vocab4}.",
            "Through the lens of {vocab3}, we see how {vocab5} violations connect to broader systems of oppression.",
            "Challenging {vocab4} requires recognizing the {vocab1} inherent in these policies. {vocab5} must be central to our analysis.",
        ]
    },
]

EVENTS = [
    "settlement expansion announcement",
    "military operation in Gaza",
    "diplomatic summit breakdown",
    "Temple Mount tensions",
    "judicial reform protests",
    "border escalation",
    "UN resolution debate",
    "prisoner release negotiations",
]

LOCATIONS = [
    "Judea and Samaria",
    "the West Bank",
    "Area C",
    "East Jerusalem",
    "Gaza border",
    "the Jordan Valley",
]

AUTHORS_BY_CLUSTER = {
    "Hardline Religious Zionist": ["Rabbi Yitzchak Ben-David", "Rav Shlomo Aviner", "Naftali Stern"],
    "Evangelical Zionist": ["Pastor John Hagee", "Dr. Samuel Cohen", "Rev. Michael Evans"],
    "Security Realist": ["Dr. Avi Cohen", "Col. (ret.) Yossi Melman", "Amir Bohbot"],
    "Liberal Zionist": ["Dr. Elena Goldberg", "Gideon Levy", "Amira Hass"],
    "Secular Palestinian Nationalist": ["Dr. Ahmed Mansour", "Hanan Ashrawi", "Saeb Erekat"],
    "Islamist Resistance": ["Sheikh Abdullah al-Gharib", "Dr. Mahmoud al-Zahar", "Khaled Meshaal"],
    "Arab Pro-Palestine": ["Dr. Khalil Al-Sayed", "Marwan Bishara", "Jamal Rayyan"],
    "Western Leftist": ["Dr. Sarah Jenkins", "Noam Chomsky", "Norman Finkelstein"],
}

def generate_article(cluster_idx: int, article_num: int) -> dict:
    """Generate a single synthetic article for testing."""
    cluster = CLUSTERS[cluster_idx]
    event = random.choice(EVENTS)
    location = random.choice(LOCATIONS)

    # Build vocab mapping
    vocab = cluster["vocab"]
    vocab_map = {f"vocab{i+1}": vocab[i % len(vocab)] for i in range(5)}
    vocab_map["event"] = event
    vocab_map["location"] = location

    # Generate content from templates
    template = random.choice(cluster["templates"])
    paragraphs = []

    for _ in range(random.randint(4, 6)):
        template = random.choice(cluster["templates"])
        para = template.format(**vocab_map)
        paragraphs.append(para)

    content = "\n\n".join(paragraphs)

    # Generate metadata
    start_date = datetime(2022, 1, 1)
    random_days = random.randint(0, 365 * 2)
    article_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

    author = random.choice(AUTHORS_BY_CLUSTER[cluster["label"]])

    title_templates = [
        f"{event.title()}: A {cluster['label']} Perspective",
        f"Analysis: {event.title()} and Its Implications",
        f"The {event.title()} Crisis Deepens",
        f"Understanding the {event.title()}",
    ]

    return {
        "title": random.choice(title_templates),
        "publication": cluster["pub"],
        "author": author,
        "timestamp": article_date,
        "event_id": event[:5].replace(" ", ""),
        "perspective_tag": cluster["label"],
        "perspective_type": cluster["type"],
        "content": content,
    }

def main():
    """Generate synthetic test corpus."""
    output_file = Path(__file__).parent / "high_quality_articles.jsonl"

    # Generate 10 articles per cluster = 80 total
    articles_per_cluster = 10

    print(f"Generating {articles_per_cluster * 8} synthetic articles...")

    articles = []
    for cluster_idx in range(8):
        for i in range(articles_per_cluster):
            article = generate_article(cluster_idx, i)
            articles.append(article)

    # Shuffle to mix clusters
    random.seed(42)
    random.shuffle(articles)

    # Write to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for article in articles:
            json.dump(article, f, ensure_ascii=False)
            f.write("\n")

    print(f"Written {len(articles)} articles to {output_file}")

    # Print distribution
    from collections import Counter
    dist = Counter(a["perspective_tag"] for a in articles)
    print("\nCluster distribution:")
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count}")

if __name__ == "__main__":
    main()
