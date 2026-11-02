"""Bind ranker weights to lexical semantics, beyond the ordered column names."""
import json
from pathlib import Path

import numpy as np

from subtlex_mixture import RECIPE, digest, validate_configuration


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or recipe.get('name') != RECIPE:
        raise ValueError('Features/models require the subtlex_mix_v1 recipe; rebuild older artifacts')
    validate_configuration(recipe)
    return recipe


def bundle_recipe(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json'].item())) if 'metadata_json' in data.files else {}
    return validate_recipe(metadata.get('feature_recipe'))


def bundle_embedding_root(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json'].item())) if 'metadata_json' in data.files else {}
    return Path(metadata.get('embedding_cache_path', Path(path).resolve().parent / 'embeddings'))


def sidecar_path(model):
    return Path(model).with_suffix(Path(model).suffix + '.recipe.json')


def save_model_recipe(model, recipe):
    validate_recipe(recipe)
    target = sidecar_path(model)
    temporary = target.with_suffix(target.suffix + '.tmp')
    try:
        temporary.write_text(json.dumps({'schema_version': 1, 'model_sha256': digest(model),
                                        'feature_recipe': recipe}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def check_model_recipe(model, recipe):
    validate_recipe(recipe)
    path = sidecar_path(model)
    if not path.is_file():
        raise ValueError('SUBTLEX model/bundle recipe requires a matching model sidecar')
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('schema_version') != 1 or record.get('model_sha256') != digest(model):
        raise ValueError('Stale model recipe sidecar hash')
    if validate_recipe(record.get('feature_recipe')) != recipe:
        raise ValueError('Incompatible model/bundle feature recipe (including alpha/source hash)')
    return path
