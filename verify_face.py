import os
import sys
import json
import warnings
import contextlib
import tempfile
import requests
import traceback
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from io import BytesIO
from functools import lru_cache

# Environment settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "0"
warnings.filterwarnings("ignore")

@contextlib.contextmanager
def suppress_tf_warnings():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            yield

def enhance_image(image, enhancement_level='standard'):
    """Apply targeted enhancements optimized for face recognition."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Auto contrast adjustment for dynamic range
    image = ImageOps.autocontrast(image, cutoff=2)
    
    if enhancement_level == 'aggressive':
        image = ImageEnhance.Contrast(image).enhance(1.5)
        image = ImageEnhance.Sharpness(image).enhance(2.0)
        image = ImageEnhance.Brightness(image).enhance(1.2)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
    elif enhancement_level == 'balanced':
        image = ImageEnhance.Contrast(image).enhance(1.3)
        image = ImageEnhance.Sharpness(image).enhance(1.7)
        image = ImageEnhance.Brightness(image).enhance(1.1)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    elif enhancement_level == 'facenet_optimized':
        image = ImageEnhance.Contrast(image).enhance(1.35)
        image = ImageEnhance.Sharpness(image).enhance(1.85)
        image = ImageEnhance.Brightness(image).enhance(1.05)
        image = image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=140, threshold=2))
    else:  # standard
        image = ImageEnhance.Contrast(image).enhance(1.3)
        image = ImageEnhance.Sharpness(image).enhance(1.7)
        image = ImageEnhance.Brightness(image).enhance(1.1)
    
    return image

@lru_cache(maxsize=16)
def load_image(path_or_url, enhanced=True, enhancement_level='standard'):
    """Load an image from a URL or local path, apply enhancement, and return the image object."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        try:
            response = requests.get(path_or_url, timeout=15)
            if response.status_code != 200:
                raise ValueError(f"Failed to download image from URL: {response.status_code}")
            img = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error downloading image: {str(e)}")
    else:
        if not os.path.exists(path_or_url):
            raise ValueError(f"File does not exist: {path_or_url}")
        try:
            img = Image.open(path_or_url).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error opening image: {str(e)}")
    
    # Resize only if necessary
    original_size = img.size
    if min(original_size) < 250:
        scale_factor = 250 / min(original_size)
        new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)
    elif max(original_size) > 1800:
        scale_factor = 1800 / max(original_size)
        new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)
    
    if enhanced:
        img = enhance_image(img, enhancement_level)
    
    return img

def save_image(img, suffix):
    """Save a PIL image to a temporary file."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    img.save(temp_file.name, format="PNG", optimize=True, quality=95)
    return temp_file.name

def to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(item) for item in obj]
    else:
        return obj

@lru_cache(maxsize=32)
def get_embedding(image_path, model_name, detector_backend):
    """Cache the embedding extraction to avoid re-computation for the same image/model combination."""
    from deepface import DeepFace
    return DeepFace.represent(
        img_path=image_path,
        model_name=model_name,
        enforce_detection=True,
        detector_backend=detector_backend,
        align=True
    )[0]["embedding"]

def compute_match_percentage(embedding1, embedding2, model_name="Facenet512"):
    """Calculate similarity score (match percentage) using cosine similarity and model-specific scaling."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    # Normalize embeddings
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    if model_name == "Facenet512":
        adjusted = ((similarity ** 0.55) * 115)
    elif model_name == "ArcFace":
        adjusted = ((similarity ** 0.60) * 112)
    else:
        adjusted = ((similarity ** 0.65) * 110)
    adjusted = min(adjusted, 100.0)
    return float(round(adjusted, 2))

def main():
    temp_files = []
    
    try:
        if len(sys.argv) < 3:
            raise ValueError("Not enough arguments. Usage: python verify_face.py <input_image_path> <reference_image_url>")
        
        input_arg = sys.argv[1]
        ref_arg = sys.argv[2]
        
        # Load images (as PIL images)
        input_img = load_image(input_arg, enhanced=True, enhancement_level='facenet_optimized')
        ref_img = load_image(ref_arg, enhanced=True, enhancement_level='facenet_optimized')
        
        # Save enhanced images as temporary files for embedding extraction
        facenet_input = save_image(input_img, "_input_facenet_optimized.png")
        facenet_ref = save_image(ref_img, "_ref_facenet_optimized.png")
        temp_files.extend([facenet_input, facenet_ref])
        
        # We'll use only two models: Facenet512 and ArcFace.
        model_weights = {
            "Facenet512": 1.2, 
            "ArcFace": 0.8
        }
        
        # Use a fixed and prioritized detector backend (you can experiment with "retinaface" or "mtcnn")
        detectors = ["retinaface", "mtcnn", "opencv"]
        
        model_results = []
        for model_name in model_weights:
            best_score = 0
            best_detector = None
            for detector in detectors:
                try:
                    # Get cached embeddings for input and reference images
                    input_embedding = get_embedding(facenet_input, model_name, detector)
                    ref_embedding = get_embedding(facenet_ref, model_name, detector)
                    score = compute_match_percentage(input_embedding, ref_embedding, model_name)
                    if score > best_score:
                        best_score = score
                        best_detector = detector
                    # Break if a sufficiently high score is achieved
                    if best_score >= 80:
                        break
                except Exception:
                    continue
            if best_score > 0:
                model_results.append((model_name, best_score, model_weights[model_name], best_detector))
        
        if not model_results:
            raise ValueError("All face recognition attempts failed. Ensure clear faces are provided.")
        
        # Use weighted average from the two models
        total_weight = sum(item[2] for item in model_results)
        weighted_score = sum(item[1] * item[2] for item in model_results) / total_weight
        weighted_score = float(round(weighted_score, 2))
        
        model_scores = {model: float(score) for model, score, _, _ in model_results}
        
        # Define a threshold for verification (you can adjust as needed)
        final_threshold = 80
        verified = weighted_score >= final_threshold
        
        # Since age and gender are optional, attempt analysis but do not fail if it takes too long
        try:
            from deepface import DeepFace
            analysis = DeepFace.analyze(
                img_path=facenet_input,
                actions=["age", "gender"],
                enforce_detection=True,
                detector_backend="retinaface",
            )[0]
            age = float(analysis["age"])
            age_margin = 3 if age > 30 else 5
            age_range = f"{max(0, int(age - age_margin))} - {int(age + age_margin)}"
            
            gender_result = {}
            if isinstance(analysis["gender"], dict):
                gender_result = {k: float(v) for k, v in analysis["gender"].items()}
            else:
                dominant = analysis["gender"]
                gender_result = {"Woman": 0.01, "Man": 0.01}
                gender_result[dominant] = 99.99
        except Exception:
            age_range = "N/A"
            gender_result = {}
        
        output = {
            "match_percentage": weighted_score,
            "verified": verified,
            "confidence": "very high" if weighted_score >= (final_threshold + 15) else "high" if weighted_score >= (final_threshold + 10) else "medium" if weighted_score >= final_threshold else "low",
            "threshold_used": final_threshold,
            "model_used": "ensemble",
            "model_results": model_scores,
            "age_range": age_range,
            "gender": gender_result
        }

        # output = {
        #     "match_percentage": weighted_score,
        #     "verified": verified,
        #     "age": int(float(age)) if isinstance(age, (int, float, str)) and age != "N/A" else "unknown",
        #     "gender": "male" if gender_result.get("Man", 0) > gender_result.get("Woman", 0) else "female"
        # }
        
        output = to_json_serializable(output)
        print(json.dumps(output))
        
    except Exception as e:
        error_details = {
            "error": "Face verification failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_details))
        sys.exit(1)
    finally:
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except:
                pass

if __name__ == "__main__":
    main()
