# core/advisor.py

import json
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class Recommendation:
    id: str
    severity: str  # critical, warning, info
    title: str
    description: str
    params: Dict[str, Any]

class AnalysisAdvisor:
    """
    Evaluates analysis metrics against a JSON knowledge base
    to provide actionable recommendations.
    """
    
    def __init__(self, kb_path: str = "core/knowledge_base.json"):
        self.rules = []
        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.rules = data.get("rules", [])
            except Exception as e:
                print(f"Error loading knowledge base: {e}")
        else:
            print(f"Knowledge base not found at {kb_path}")

    def evaluate(self, metrics: Dict[str, Any]) -> List[Recommendation]:
        """
        Runs the rule engine against the provided metrics.
        """
        recommendations = []
        
        # Safe access to block energy
        energy = metrics.get("block_energy", {})
        e_in = energy.get("IN", 0.0)
        e_mid = energy.get("MID", 0.0)
        e_out = energy.get("OUT", 0.0)
        
        # Calculate derived metrics for PSP
        # mid_dominance: core conceptual strength vs structural/style constraints
        denominator = (e_in + e_out)
        mid_dominance = e_mid / denominator if denominator > 0 else 1.0

        # Safe context for evaluation
        context = {
            "kurtosis": metrics.get("kurtosis", 0.0),
            "energy": energy,
            "mid_dominance": mid_dominance,
            "current_rank": metrics.get("current_rank", 0),
            "current_alpha": metrics.get("current_alpha", 0),
            "magnitude": metrics.get("magnitude", 0.0),
            "knee_rank": metrics.get("knee_rank", 0)
        }

        for rule in self.rules:
            try:
                condition = rule.get("condition", "False")
                if eval(condition, {"__builtins__": {}}, context):
                    
                    final_params = rule.get("suggested_params", {}).copy()
                    for k, v in final_params.items():
                        if isinstance(v, str) and v in context:
                            final_params[k] = context[v]

                    rec = Recommendation(
                        id=rule["id"],
                        severity=rule["severity"],
                        title=rule["title"],
                        description=rule["description"],
                        params=final_params
                    )
                    recommendations.append(rec)
            except Exception as e:
                print(f"Error evaluating rule {rule.get('id')}: {e}")
                
        return recommendations