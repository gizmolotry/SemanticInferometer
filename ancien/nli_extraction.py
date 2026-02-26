"""
FIXED Multi-Framing NLI Extraction - Using Logits

Key change: Uses NLI logits [contradiction, neutral, entailment] instead of CLS embeddings.
This preserves semantic polarity and distinguishes opposite framings.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Union
import yaml
from pathlib import Path


class MultiFramingNLIExtractorLogits(nn.Module):
    """
    Extract multi-perspective representations using NLI logits.
    
    Each framing produces a 3D vector: [contradiction, neutral, entailment]
    Total output: n_framings * 3 dimensions
    
    This preserves semantic polarity better than CLS embeddings.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v2-xlarge-mnli",
        queries_config: str = "config/framing_queries.yaml",
        device: str = "cuda",
        max_length: int = 512,
    ):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Load NLI model WITH classification head
        print(f"Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nli_model.to(self.device)
        self.nli_model.eval()

        for param in self.nli_model.parameters():
            param.requires_grad = False

        # Load framing queries
        self.framing_queries = self._load_queries(queries_config)
        self.n_framings = len(self.framing_queries)
        self.logits_dim = 3  # [contradiction, neutral, entailment]

        print(f"Initialized extractor with {self.n_framings} framing perspectives")
        print(f"Logits per framing: {self.logits_dim}")
        print(f"Total feature dimension: {self.n_framings * self.logits_dim}")
        print(f"Using NLI LOGITS instead of CLS embeddings")

    def _load_queries(self, config_path: str) -> List[str]:
        """Load framing queries from YAML config."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        queries = []
        for query_name, query_data in config["queries"].items():
            queries.append(query_data["hypothesis"])

        return queries

    @torch.no_grad()
    def extract_single_framing(self, article_text: str, hypothesis: str) -> torch.Tensor:
        """
        Extract NLI logits for one framing perspective.
        
        Returns [contradiction_score, neutral_score, entailment_score]
        """
        if not isinstance(article_text, str) or len(article_text.strip()) == 0:
            raise ValueError("Invalid or empty article text.")

        inputs = self.tokenizer(
            article_text,
            hypothesis,
            return_tensors="pt",
            truncation='only_first',
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        # Get logits from classification head
        outputs = self.nli_model(**inputs)
        logits = outputs.logits.squeeze(0)  # [3] - [contradiction, neutral, entailment]
        
        # Convert to probabilities (optional, could also use raw logits)
        probs = torch.softmax(logits, dim=-1)
        
        return probs.cpu()

    @torch.no_grad()
    def extract_multi_framing(self, article_text: str) -> torch.Tensor:
        """
        Extract NLI logits across all framing perspectives.
        
        Returns flattened vector of all logits: [n_framings * 3]
        """
        all_logits = []
        for query in self.framing_queries:
            logits = self.extract_single_framing(article_text, query)
            all_logits.append(logits)

        # Stack and flatten: [n_framings, 3] -> [n_framings * 3]
        return torch.cat(all_logits, dim=-1)

    def extract_batch(
        self,
        articles: List[Union[str, Dict[str, str]]],
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Extract multi-framing logits for a batch of articles.
        """
        features = []
        iterator = articles

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(articles, desc="Extracting NLI logits")
            except ImportError:
                pass

        for idx, article in enumerate(iterator):
            try:
                if isinstance(article, dict):
                    text = article.get("content", article.get("text", ""))
                elif isinstance(article, str):
                    text = article.strip()
                else:
                    raise TypeError(
                        f"Invalid article type at index {idx}: {type(article)}"
                    )

                if not text or len(text) < 5:
                    print(f"[WARN] Skipping article {idx}: empty or too short.")
                    continue

                feat = self.extract_multi_framing(text)
                features.append(feat)

            except Exception as e:
                print(f"[ERROR] Failed to process article {idx}: {e}")
                continue

        if not features:
            raise ValueError("No valid articles processed.")

        try:
            return torch.stack(features)
        except Exception as e:
            raise RuntimeError(f"Failed to stack {len(features)} feature tensors: {e}")


if __name__ == "__main__":
    print("="*70)
    print("Testing NLI Logits Extractor")
    print("="*70)
    
    # Initialize
    extractor = MultiFramingNLIExtractorLogits()
    
    # Test articles - biased in opposite directions
    articles = [
        {
            'content': """
            Hamas terrorists launched unprovoked rocket attacks on innocent Israeli civilians.
            Israel has the right to defend itself against these barbaric attacks.
            The IDF takes every precaution to avoid civilian casualties.
            """
        },
        {
            'content': """
            Israeli forces conducted brutal attacks on Palestinian civilians in Gaza.
            The resistance fighters are defending their homeland against occupation.
            Israel deliberately targets schools and hospitals.
            """
        }
    ]
    
    # Extract
    features = extractor.extract_batch(articles, show_progress=False)
    
    print(f"\nExtracted features shape: {features.shape}")
    print(f"Expected: ({len(articles)}, {extractor.n_framings * 3})")
    
    # Compare first two framings (should be opposites)
    print(f"\nArticle 0 (pro-Israel):")
    print(f"  Frame 0 (Israel responsible): {features[0, 0:3].numpy()}")
    print(f"  Frame 1 (Hamas responsible): {features[0, 3:6].numpy()}")
    
    print(f"\nArticle 1 (pro-Hamas):")
    print(f"  Frame 0 (Israel responsible): {features[1, 0:3].numpy()}")
    print(f"  Frame 1 (Hamas responsible): {features[1, 3:6].numpy()}")
    
    # Compute cosine similarity
    from torch.nn.functional import cosine_similarity
    
    feat0_norm = features[0] / features[0].norm()
    feat1_norm = features[1] / features[1].norm()
    similarity = (feat0_norm @ feat1_norm).item()
    
    print(f"\nCross-article similarity: {similarity:.4f}")
    print(f"Expected: <0.5 (opposite biases)")
    
    if similarity < 0.5:
        print("✓ SUCCESS: Opposite articles are distinguishable!")
    else:
        print("⚠ Articles still too similar")
    
    print("\n" + "="*70)
    MultiFramingNLIExtractor = MultiFramingNLIExtractorLogits