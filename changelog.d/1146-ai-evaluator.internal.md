- **`quality/ai_evaluator.py` logs with lazy `%`-style arguments (#1146).**
  Its four multi-line f-string logger calls in the DeBERTa / MS-MARCO
  fallback path now pass the scores (`%.3f`), thresholds and decision (`%s`)
  as arguments. They were invisible to the line gate and only the AST scan
  caught them. The module is added to `GUARDED_MODULES`.
