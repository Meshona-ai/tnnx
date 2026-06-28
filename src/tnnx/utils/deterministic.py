def stable_dict[V](mapping: dict[str, V]) -> dict[str, V]:
    return {key: mapping[key] for key in sorted(mapping.keys())}
