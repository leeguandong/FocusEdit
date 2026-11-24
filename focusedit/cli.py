"""Edit one image with FocusEdit."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--source-prompt", required=True)
    p.add_argument("--target-prompt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--baseline", action="store_true", help="Run the unmodulated base editor")
    args = p.parse_args()
    from PIL import Image
    from .pipeline import FocusEditPipeline
    cfg = json.loads(Path(args.config).read_text()) if args.config else None
    pipe = FocusEditPipeline.from_pretrained(args.model, config=cfg, device=args.device)
    result = pipe(Image.open(args.image), args.source_prompt, args.target_prompt, baseline=args.baseline)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(dest)
    if result.mask is not None:
        result.mask.save(dest.with_name(dest.stem + "_mask.png"))


if __name__ == "__main__":
    main()
