import json
import os
import argparse
import time
from typing import Literal
import google.generativeai as genai

def format_thoughts(thoughts_list, thought_type: Literal["top_k", "top_p"]):
    """Format the top-k or top-p thoughts for display."""
    formatted = []
    for i, thought in enumerate(thoughts_list):
        thought_str = f"Latent thought {i+1}:\n"
        for item in thought[:20]:  # Limit to 20 tokens for readability
            thought_str += f"  {item['token']}: {item['prob']*100:.1f}%\n"
        if len(thought) > 20:
            thought_str += f"  ... and {len(thought) - 20} more tokens\n"
        formatted.append(thought_str)
    return "\n".join(formatted)

def create_prompt(question, reasoning, include_thoughts=None, thoughts_data=None):
    """Create the prompt for Gemini."""
    prompt = f"""You are analyzing a language model's reasoning on a math problem.

Question: {question}

Model's reasoning (excluding final answer): {reasoning}
"""

    if include_thoughts and thoughts_data:
        if include_thoughts == "top_k":
            prompt += f"\n\nTop-5 token representations of the model's continuous thoughts:\n"
            prompt += format_thoughts(thoughts_data["top_k_thoughts"], "top_k")
        elif include_thoughts == "top_p":
            prompt += f"\n\nTop-p (p=0.9) token representations of the model's continuous thoughts:\n"
            prompt += format_thoughts(thoughts_data["top_p_thoughts"], "top_p")
        elif include_thoughts == "both":
            prompt += f"\n\nTop-5 token representations of the model's continuous thoughts:\n"
            prompt += format_thoughts(thoughts_data["top_k_thoughts"], "top_k")
            prompt += f"\n\nTop-p (p=0.9) token representations:\n"
            prompt += format_thoughts(thoughts_data["top_p_thoughts"], "top_p")

    prompt += """

Based on the reasoning provided (and the continuous thought representations if provided), predict whether the model answered this math problem CORRECTLY or INCORRECTLY.

Respond in the following JSON format:
{
  "prediction": "correct" or "incorrect",
  "confidence": <float between 0 and 1>,
  "explanation": "<brief explanation of your prediction>"
}
"""

    return prompt

def main():
    parser = argparse.ArgumentParser(description="Analyze Coconut's reasoning using Gemini API")
    parser.add_argument("--input", default="legibility_results.json",
                        help="Input JSON file with legibility results")
    parser.add_argument("--output", default="gemini_analysis.json",
                        help="Output JSON file for Gemini's analysis")
    parser.add_argument("--include-thoughts", choices=["none", "top_k", "top_p", "both"],
                        default="none",
                        help="Whether to include thought representations in the prompt")
    parser.add_argument("--api-key", default=None,
                        help="Gemini API key (if not set, will use GOOGLE_API_KEY env var)")
    args = parser.parse_args()

    # Set up Gemini API
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please provide API key via --api-key or GOOGLE_API_KEY environment variable")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    # Load legibility results
    print(f"Loading results from {args.input}...")
    with open(args.input, "r") as f:
        legibility_results = json.load(f)

    print(f"Processing {len(legibility_results)} questions...")

    analysis_results = []
    correct_predictions = 0
    total_questions = len(legibility_results)

    for i, result in enumerate(legibility_results):
        print(f"\nProcessing question {i+1}/{total_questions}...")

        question = result["question"]
        reasoning = result["model_reasoning"]
        actual_correctness = result["is_correct"]

        # Create prompt
        include_thoughts = None if args.include_thoughts == "none" else args.include_thoughts
        prompt = create_prompt(question, reasoning, include_thoughts, result)

        # Call Gemini API with retry logic
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )
                response_text = response.text

                # Parse JSON response
                # Remove markdown code blocks if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()

                gemini_response = json.loads(response_text)

                # Check if prediction is correct
                predicted_correct = gemini_response["prediction"].lower() == "correct"
                prediction_is_correct = predicted_correct == actual_correctness

                if prediction_is_correct:
                    correct_predictions += 1

                # Store analysis result
                analysis_result = {
                    "question_idx": result["question_idx"],
                    "question": question,
                    "model_reasoning": reasoning,
                    "actual_correctness": actual_correctness,
                    "gemini_prediction": gemini_response["prediction"],
                    "gemini_confidence": gemini_response["confidence"],
                    "gemini_explanation": gemini_response["explanation"],
                    "prediction_is_correct": prediction_is_correct
                }
                analysis_results.append(analysis_result)

                print(f"  Actual: {'correct' if actual_correctness else 'incorrect'}")
                print(f"  Gemini predicted: {gemini_response['prediction']} (confidence: {gemini_response['confidence']:.2f})")
                print(f"  Prediction correct: {prediction_is_correct}")

                # Success, break out of retry loop
                break

            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"  JSON decode error on attempt {attempt+1}/{max_retries}, retrying...")
                    print(f"  Response was: {response_text[:200]}...")
                    time.sleep(retry_delay)
                else:
                    print(f"  Error after {max_retries} attempts: {e}")
                    print(f"  Response was: {response_text[:200]}...")
                    analysis_result = {
                        "question_idx": result["question_idx"],
                        "question": question,
                        "model_reasoning": reasoning,
                        "actual_correctness": actual_correctness,
                        "error": f"JSON decode error after {max_retries} attempts: {str(e)}",
                        "raw_response": response_text
                    }
                    analysis_results.append(analysis_result)

            except Exception as e:
                if "API_KEY_INVALID" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        print(f"  API error on attempt {attempt+1}/{max_retries}: {e}")
                        print(f"  Waiting {retry_delay * (attempt + 1)} seconds before retry...")
                        time.sleep(retry_delay * (attempt + 1))
                    else:
                        print(f"  Error after {max_retries} attempts: {e}")
                        analysis_result = {
                            "question_idx": result["question_idx"],
                            "question": question,
                            "model_reasoning": reasoning,
                            "actual_correctness": actual_correctness,
                            "error": str(e)
                        }
                        analysis_results.append(analysis_result)
                else:
                    print(f"  Error processing question {i+1}: {e}")
                    analysis_result = {
                        "question_idx": result["question_idx"],
                        "question": question,
                        "model_reasoning": reasoning,
                        "actual_correctness": actual_correctness,
                        "error": str(e)
                    }
                    analysis_results.append(analysis_result)
                    break

        # Small delay between API calls to avoid rate limiting
        time.sleep(0.5)

    # Calculate accuracy (only for successfully processed questions)
    successfully_processed = sum(1 for r in analysis_results if "error" not in r)
    accuracy = correct_predictions / successfully_processed if successfully_processed > 0 else 0
    error_count = total_questions - successfully_processed

    # Save results
    output_data = {
        "config": {
            "input_file": args.input,
            "include_thoughts": args.include_thoughts,
            "model": "gemini-2.5-flash-lite"
        },
        "summary": {
            "total_questions": total_questions,
            "successfully_processed": successfully_processed,
            "errors": error_count,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy
        },
        "results": analysis_results
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"Total questions: {total_questions}")
    print(f"Successfully processed: {successfully_processed}")
    print(f"Errors: {error_count}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results saved to {args.output}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
