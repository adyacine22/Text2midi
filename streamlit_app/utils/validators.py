"""Input validation utilities"""

import re


def validate_prompt(prompt, max_length=500):
    """
    Validate user prompt

    Args:
        prompt: User input text
        max_length: Maximum allowed length

    Returns:
        tuple: (is_valid, error_message)
    """
    # Check if empty
    if not prompt or not prompt.strip():
        return False, "Please enter a musical description"

    # Check length
    if len(prompt) > max_length:
        return (
            False,
            f"Description is too long ({len(prompt)} characters). Please keep it under {max_length} characters.",
        )

    # Check for minimum length
    if len(prompt.strip()) < 10:
        return (
            False,
            "Description is too short. Please provide more details (at least 10 characters).",
        )

    return True, None


def sanitize_prompt(prompt):
    """
    Sanitize user input by removing potentially problematic characters

    Args:
        prompt: User input text

    Returns:
        str: Sanitized prompt
    """
    # Remove leading/trailing whitespace
    prompt = prompt.strip()

    # Replace multiple spaces with single space
    prompt = re.sub(r"\s+", " ", prompt)

    # Remove control characters except newlines
    prompt = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]", "", prompt)

    return prompt


def validate_generation_params(max_length, temperature):
    """
    Validate generation parameters

    Args:
        max_length: Maximum sequence length
        temperature: Sampling temperature

    Returns:
        tuple: (is_valid, error_message)
    """
    # Check max_length
    if not (100 <= max_length <= 2048):
        return False, "Max length must be between 100 and 2048"

    # Check temperature
    if not (0.1 <= temperature <= 2.0):
        return False, "Temperature must be between 0.1 and 2.0"

    return True, None
