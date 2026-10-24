"""Procedural decoding workloads; these do not measure answer quality."""

from __future__ import annotations

import hashlib
import json
import random

from dart.data.corpus import SPLITS, Prompt, validate_records

VERSION = "dart-procedural-v1"
FAMILIES = {
    "instruction": {
        "train": ("explain_processing", "plan_work", "summarize_reports"),
        "validation": ("compare_options", "rewrite_message"),
        "test": ("answer_from_records", "extract_actions"),
    },
    "code": {
        "train": ("sum_above_threshold", "clean_words", "lookup_record"),
        "validation": ("flatten_lists", "count_characters"),
        "test": ("balanced_brackets", "merge_intervals"),
    },
    "structured": {
        "train": ("records_to_json", "records_to_csv", "yaml_config"),
        "validation": ("tool_arguments", "markdown_table"),
        "test": ("json_schema", "xml_records"),
    },
    "repetition": {
        "train": ("copy_sentences", "word_cycle", "repeat_key_values"),
        "validation": ("echo_letter", "symbol_cycle"),
        "test": ("copy_grid", "numbered_refrain"),
    },
}


def _content(workload: str, family: str, index: int, rng: random.Random) -> tuple[str, str, str]:
    size = (2, 8, 24)[index % 3]
    colors = ("red", "blue", "green", "gold", "silver", "white", "purple", "orange")
    names = [f"{rng.choice(colors)}-{rng.choice(('fern', 'oak', 'pine', 'reed'))}-{100 * i + rng.randrange(100)}" for i in range(size)]
    numbers = [rng.randrange(10, 990) for _ in range(size)]
    facts = "\n".join(f"{name} processed {value} orders in {1 + value % 8} hours." for name, value in zip(names, numbers))
    table = "\n".join(f"{name};{value};{str(value % 2 == 0).lower()}" for name, value in zip(names, numbers))

    if workload == "instruction":
        content = {
            "explain_processing": (f"Explain how a {rng.choice(('loop', 'list', 'dictionary'))} could help a new programmer process these reports.", facts),
            "plan_work": (f"Write {2 + index % 4} steps to organise next week's work using these warehouse reports.", facts),
            "summarize_reports": ("Summarize these reports in two sentences. Mention the main differences.", facts),
            "compare_options": ("Compare these options. State one advantage and one limitation for each.", f"Option A costs {numbers[0]} dollars and handles {numbers[1]} orders per day.\nOption B costs {numbers[1]} dollars and handles {numbers[0]} orders per day."),
            "rewrite_message": ("Rewrite this message as a polite professional email.", "hey team, here is what happened today:\n" + facts + "\nplease sort this out soon"),
            "answer_from_records": (f"How many orders did {names[-1]} process? Answer using only the records.", facts),
            "extract_actions": ("Extract the action items as a numbered list.", "\n".join(f"Meeting note: {name} must pack {value} orders by Friday." for name, value in zip(names, numbers))),
        }
        text, context = content[family]
        return text, context, "supra"

    if workload == "code":
        definitions = {
            "sum_above_threshold": (f"values, threshold={numbers[0]}", "Return the sum of values greater than threshold.", numbers),
            "clean_words": ("words", "Strip whitespace, lowercase words, and remove duplicates while preserving order.", names),
            "lookup_record": (f"records, key='{names[0]}'", "Return the first record with this name, or None.", [{"name": name, "count": value} for name, value in zip(names, numbers)]),
            "flatten_lists": ("rows", "Flatten a list of lists by one level.", [numbers[i:i + 2] for i in range(0, size, 2)]),
            "count_characters": ("text", "Return a dictionary counting each character except whitespace.", " ".join(names)),
            "balanced_brackets": ("text", "Return whether all round and square brackets are balanced and correctly nested.", "".join(f"([{name}])" for name in names)),
            "merge_intervals": ("intervals", "Merge overlapping closed intervals and return them sorted by start.", [[value, value + 5 + i] for i, value in enumerate(numbers)]),
        }
        arguments, description, example = definitions[family]
        text = f"# Example input: {json.dumps(example)}\ndef {family}({arguments}):\n    \"\"\"{description}\"\"\"\n    "
        return text, "", "raw"

    if workload == "structured":
        content = {
            "records_to_json": ("Convert these semicolon-separated rows to a JSON array. Fields: name (string), count (integer), active (boolean). Return JSON only.", table),
            "records_to_csv": ("Convert these rows to CSV with header name,count,active. Preserve row order.", table),
            "yaml_config": ("Write a YAML configuration with a jobs list. Each job needs name, retries, and enabled fields.", table),
            "tool_arguments": ("Emit a JSON tool call with tool='search' and arguments containing query and top_k.", f"Search for shipping reports about {names[0]} and {names[1]}. Return the top {1 + numbers[0] % 8} results."),
            "markdown_table": ("Make a Markdown table with columns Product, Quantity, Available.", table),
            "json_schema": ("Write a JSON Schema for this specification. Return JSON only.", "An object with these required integer fields: " + ", ".join(names)),
            "xml_records": ("Convert these records to XML using a records root and record children with name, count, and active elements.", table),
        }
        text, context = content[family]
        return text, context, "supra"

    if family == "copy_sentences":
        return "Copy these sentences exactly once. Do not add a heading.", facts, "supra"
    if family == "word_cycle":
        cycle = " ".join(names[:2]) + " "
        return "Continue the repeating sequence:\n" + cycle * size, "", "raw"
    if family == "repeat_key_values":
        return "Repeat these key=value lines exactly in the same order.", "\n".join(f"{name}={value}" for name, value in zip(names, numbers)), "supra"
    if family == "echo_letter":
        return "Quote the complete letter verbatim, including its greeting and signature.", f"Dear {names[0]},\n{facts}\nRegards,\n{names[-1]}", "supra"
    if family == "symbol_cycle":
        pattern = f"[{numbers[0]}]({numbers[1]})"
        return "Continue this pattern without commentary:\n" + pattern * size, "", "raw"
    if family == "copy_grid":
        return "Copy this tab-separated grid exactly. Keep the tabs and line breaks.", table.replace(";", "\t"), "supra"
    if family == "numbered_refrain":
        return "Continue the numbered lines:\n" + "".join(f"{i + 1}. {names[0]} carries {numbers[0]} boxes.\n" for i in range(size)), "", "raw"
    raise ValueError(f"unknown family: {family}")


def build_pilot(seed: int = 17, train_per_family: int = 16, validation_per_family: int = 8, test_per_family: int = 16) -> list[Prompt]:
    counts = dict(zip(SPLITS, (train_per_family, validation_per_family, test_per_family)))
    if type(seed) is not int or any(type(count) is not int or count < 1 for count in counts.values()):
        raise ValueError("use an integer seed and positive counts per family")
    total = sum(len(groups[split]) * count for groups in FAMILIES.values() for split, count in counts.items())
    if total > 10_000:
        raise ValueError("pilot exceeds the 10000-record limit")
    records = []
    for workload, groups in FAMILIES.items():
        for split, families in groups.items():
            for family in families:
                for index in range(counts[split]):
                    key = f"{VERSION}:{seed}:{family}:{index}"
                    rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))
                    text, context, format_name = _content(workload, family, index, rng)
                    records.append(Prompt(
                        id=f"v1-{family}-{index:06d}", source=f"generated:{family}:{index:06d}",
                        family=family, workload=workload, split=split, text=text, input=context,
                        format=format_name, max_new_tokens=(32, 64, 128)[(index // 3) % 3],
                    ))
    validate_records(records)
    return records
