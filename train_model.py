"""
Model Training for Hand Gesture Recognition
Phase 5: Train and evaluate ML classifiers on collected landmark data

=============================================================================
MODEL SELECTION DECISIONS
=============================================================================

1. PRIMARY MODEL: Random Forest
   - Why: Fast training, no hyperparameter sensitivity, handles multiclass 
     natively, excellent for tabular data like landmarks
   - Training: O(n * m * log(n)) where n=samples, m=features
   - Inference: O(tree_depth * n_trees) - very fast, just tree traversals
   - Multiclass: Native support (no OvO/OvA needed)

2. SECONDARY MODEL: SVM with RBF Kernel
   - Why: Strong performance on small-medium datasets, good generalization
   - Multiclass Strategy: One-vs-One (OvO) - scikit-learn default for SVC
     * OvO trains k*(k-1)/2 binary classifiers (45 for 10 classes)
     * Pros: Each classifier sees balanced classes, works well for <100k samples
     * Cons: More classifiers than OvA, but each trains on subset
   - Why not One-vs-All (OvA)?
     * OvA trains k classifiers, but each sees imbalanced data (1 class vs all)
     * OvO typically performs better for gesture recognition

3. WHY NOT NEURAL NETWORKS?
   - Dataset size: Landmark data is small (1000s of samples), DNNs need more
   - Feature space: 63 features (21 landmarks * 3 coords) - not high-dimensional
   - Interpretability: RF/SVM easier to debug
   - Deployment: No GPU/framework dependencies

4. FEATURE SPACE
   - Input: 63 normalized features (x, y, z for 21 landmarks)
   - Normalization: Wrist-centered, unit-scaled (done during collection)
   - No additional feature engineering needed for landmarks

=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
COMBINED_CSV = DATA_DIR / "combined" / "all_landmarks.csv"

# Page config
st.set_page_config(
    page_title="Train Gesture Model",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
    }
    .metric-card {
        background: rgba(233, 69, 96, 0.1);
        border: 1px solid #e94560;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def ensure_directories():
    """Create necessary directories."""
    MODELS_DIR.mkdir(exist_ok=True)


def load_training_data():
    """Load and prepare training data from combined CSV."""
    if not COMBINED_CSV.exists():
        return None, None, None, None
    
    df = pd.read_csv(COMBINED_CSV)
    
    # Feature columns (x0, y0, z0, x1, y1, z1, ..., x20, y20, z20)
    feature_cols = []
    for i in range(21):
        feature_cols.extend([f"x{i}", f"y{i}", f"z{i}"])
    
    # Check if all feature columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        return None, None, None, None
    
    X = df[feature_cols].values
    y = df["label"].values
    
    # Remove any rows with NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X, y, feature_cols, df


def train_random_forest(X_train, y_train, n_estimators=200, max_depth=20):
    """Train Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, C=1.0, gamma='scale'):
    """
    Train SVM classifier with RBF kernel.
    Uses One-vs-One (OvO) strategy (scikit-learn default for SVC).
    """
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = SVC(
        C=C,
        kernel='rbf',
        gamma=gamma,
        decision_function_shape='ovr',  # Output shape, actual training is OvO
        class_weight='balanced',
        probability=True,  # Enable probability estimates
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def evaluate_model(model, X_test, y_test, scaler=None):
    """Evaluate model and return metrics."""
    if scaler is not None:
        X_test = scaler.transform(X_test)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': y_pred
    }


def cross_validate_model(model, X, y, cv=5, scaler=None):
    """Perform cross-validation."""
    if scaler is not None:
        X = scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    
    return {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
        'scores': scores
    }


def save_model(model, label_encoder, scaler, model_name, metrics, config):
    """Save trained model and metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODELS_DIR / f"{model_name}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, model_dir / "model.joblib")
    
    # Save label encoder
    joblib.dump(label_encoder, model_dir / "label_encoder.joblib")
    
    # Save scaler (for SVM)
    if scaler is not None:
        joblib.dump(scaler, model_dir / "scaler.joblib")
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "config": config,
        "metrics": {
            "accuracy": float(metrics['accuracy']),
            "f1_score": float(metrics['f1_score']),
        },
        "classes": label_encoder.classes_.tolist(),
        "n_classes": len(label_encoder.classes_),
        "requires_scaling": scaler is not None
    }
    
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return model_dir


def get_existing_models():
    """Get list of trained models."""
    if not MODELS_DIR.exists():
        return []
    
    models = []
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    metadata['path'] = str(model_dir)
                    models.append(metadata)
    
    return sorted(models, key=lambda x: x.get('timestamp', ''), reverse=True)


def main():
    """Main training application."""
    ensure_directories()
    
    st.title("🧠 Train Gesture Recognition Model")
    st.markdown("*Phase 5: Train ML classifiers on collected landmark data*")
    
    # Sidebar - Model Selection Documentation
    with st.sidebar:
        st.header("📚 Model Selection")
        
        with st.expander("🌲 Why Random Forest?", expanded=False):
            st.markdown("""
            **Advantages:**
            - Fast training & inference
            - No hyperparameter sensitivity
            - Handles multiclass natively
            - Works great for tabular data
            - No feature scaling needed
            
            **Settings:**
            - `n_estimators`: 200 trees
            - `max_depth`: 20 (prevent overfitting)
            - `class_weight`: balanced
            """)
        
        with st.expander("🎯 Why SVM with OvO?", expanded=False):
            st.markdown("""
            **One-vs-One (OvO):**
            - Trains k*(k-1)/2 classifiers
            - Each sees balanced binary data
            - Better for gesture recognition
            
            **Why not One-vs-All?**
            - OvA creates imbalanced problems
            - Each classifier: 1 class vs ALL others
            - OvO typically outperforms for <100k samples
            
            **Settings:**
            - `kernel`: RBF (non-linear boundaries)
            - `C`: 1.0 (regularization)
            - `probability`: True (confidence scores)
            """)
        
        with st.expander("❌ Why not Neural Networks?", expanded=False):
            st.markdown("""
            **Not ideal because:**
            - Small dataset (1000s of samples)
            - Low-dimensional features (63)
            - No GPU dependency needed
            - RF/SVM are more interpretable
            - Faster training & deployment
            """)
        
        st.divider()
        st.header("📦 Trained Models")
        
        models = get_existing_models()
        if models:
            for m in models[:5]:
                st.write(f"**{m['model_name']}**")
                st.write(f"  Accuracy: {m['metrics']['accuracy']:.1%}")
                st.write(f"  Classes: {m['n_classes']}")
                st.caption(m['timestamp'])
        else:
            st.info("No models trained yet")
    
    # Main content
    tab_train, tab_evaluate, tab_export = st.tabs(["🎓 Train", "📊 Evaluate", "📤 Export"])
    
    # Load data
    X, y, feature_cols, df = load_training_data()
    
    with tab_train:
        st.subheader("🎓 Train New Model")
        
        if X is None:
            st.warning("⚠️ No training data found. Export combined CSV from Data Collection tool first.")
            st.info(f"Expected path: `{COMBINED_CSV}`")
            return
        
        # Data overview
        st.markdown("### 📊 Training Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", f"{len(X):,}")
        with col2:
            st.metric("Features", len(feature_cols))
        with col3:
            unique_classes = np.unique(y)
            st.metric("Classes", len(unique_classes))
        with col4:
            st.metric("Sessions", df['session'].nunique() if 'session' in df.columns else "N/A")
        
        # Class distribution
        st.markdown("**Class Distribution:**")
        class_counts = pd.Series(y).value_counts().sort_index()
        st.bar_chart(class_counts)
        
        st.divider()
        
        # Training configuration
        st.markdown("### ⚙️ Training Configuration")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            model_type = st.selectbox(
                "Model Type",
                ["Random Forest", "SVM (RBF + OvO)", "Both (Compare)"],
                help="Random Forest is recommended for speed; SVM for potentially better accuracy"
            )
            
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                help="Portion of data reserved for evaluation"
            )
        
        with config_col2:
            if model_type in ["Random Forest", "Both (Compare)"]:
                n_estimators = st.slider("RF: Number of Trees", 50, 500, 200, 50)
                max_depth = st.slider("RF: Max Depth", 5, 50, 20, 5)
            
            if model_type in ["SVM (RBF + OvO)", "Both (Compare)"]:
                svm_c = st.select_slider("SVM: C (Regularization)", 
                                         options=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 
                                         value=1.0)
        
        st.divider()
        
        # Train button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            
            st.info(f"Training on {len(X_train):,} samples, testing on {len(X_test):,} samples")
            
            results = {}
            
            # Train Random Forest
            if model_type in ["Random Forest", "Both (Compare)"]:
                with st.spinner("Training Random Forest..."):
                    rf_model = train_random_forest(X_train, y_train, n_estimators, max_depth)
                    rf_metrics = evaluate_model(rf_model, X_test, y_test)
                    
                    # Save model
                    rf_config = {"n_estimators": n_estimators, "max_depth": max_depth, "test_size": test_size}
                    rf_path = save_model(rf_model, label_encoder, None, "random_forest", rf_metrics, rf_config)
                    
                    results['Random Forest'] = {
                        'model': rf_model,
                        'metrics': rf_metrics,
                        'path': rf_path
                    }
                    
                st.success(f"✅ Random Forest trained - Accuracy: {rf_metrics['accuracy']:.1%}")
            
            # Train SVM
            if model_type in ["SVM (RBF + OvO)", "Both (Compare)"]:
                with st.spinner("Training SVM (this may take longer)..."):
                    svm_model, svm_scaler = train_svm(X_train, y_train, C=svm_c)
                    svm_metrics = evaluate_model(svm_model, X_test, y_test, svm_scaler)
                    
                    # Save model
                    svm_config = {"C": svm_c, "kernel": "rbf", "strategy": "OvO", "test_size": test_size}
                    svm_path = save_model(svm_model, label_encoder, svm_scaler, "svm_ovo", svm_metrics, svm_config)
                    
                    results['SVM (OvO)'] = {
                        'model': svm_model,
                        'metrics': svm_metrics,
                        'scaler': svm_scaler,
                        'path': svm_path
                    }
                    
                st.success(f"✅ SVM trained - Accuracy: {svm_metrics['accuracy']:.1%}")
            
            # Display comparison
            if len(results) > 0:
                st.divider()
                st.markdown("### 📈 Results")
                
                for name, data in results.items():
                    with st.expander(f"**{name}** - {data['metrics']['accuracy']:.1%} accuracy", expanded=True):
                        met_col1, met_col2 = st.columns(2)
                        
                        with met_col1:
                            st.metric("Accuracy", f"{data['metrics']['accuracy']:.1%}")
                            st.metric("F1 Score (weighted)", f"{data['metrics']['f1_score']:.3f}")
                        
                        with met_col2:
                            st.write("**Per-class Performance:**")
                            report = data['metrics']['classification_report']
                            for cls in label_encoder.classes_:
                                if cls in report:
                                    cls_data = report[cls]
                                    st.write(f"  {cls}: P={cls_data['precision']:.2f}, R={cls_data['recall']:.2f}")
                        
                        st.write(f"📁 Saved to: `{data['path']}`")
                
                # Recommendation
                if len(results) == 2:
                    st.divider()
                    rf_acc = results['Random Forest']['metrics']['accuracy']
                    svm_acc = results['SVM (OvO)']['metrics']['accuracy']
                    
                    if rf_acc >= svm_acc - 0.02:  # RF within 2% of SVM
                        st.success("**Recommendation:** Use **Random Forest** - similar accuracy, faster inference")
                    else:
                        st.info(f"**Recommendation:** Use **SVM** - {(svm_acc - rf_acc)*100:.1f}% better accuracy")
    
    with tab_evaluate:
        st.subheader("📊 Model Evaluation")
        
        models = get_existing_models()
        if not models:
            st.info("No trained models to evaluate. Train a model first.")
        else:
            selected_model = st.selectbox(
                "Select Model",
                options=models,
                format_func=lambda x: f"{x['model_name']} ({x['timestamp']}) - {x['metrics']['accuracy']:.1%}"
            )
            
            if selected_model and st.button("Load & Evaluate"):
                model_path = Path(selected_model['path'])
                
                # Load model
                model = joblib.load(model_path / "model.joblib")
                label_encoder = joblib.load(model_path / "label_encoder.joblib")
                scaler = None
                if selected_model.get('requires_scaling'):
                    scaler = joblib.load(model_path / "scaler.joblib")
                
                st.success(f"Loaded model: {selected_model['model_name']}")
                
                # Show metadata
                st.json(selected_model)
    
    with tab_export:
        st.subheader("📤 Export Model for Deployment")
        
        models = get_existing_models()
        if not models:
            st.info("No trained models to export. Train a model first.")
        else:
            selected_model = st.selectbox(
                "Select Model to Export",
                options=models,
                format_func=lambda x: f"{x['model_name']} ({x['timestamp']}) - {x['metrics']['accuracy']:.1%}",
                key="export_select"
            )
            
            if selected_model:
                st.markdown(f"""
                ### Export Instructions
                
                Model saved at: `{selected_model['path']}`
                
                **Files included:**
                - `model.joblib` - The trained classifier
                - `label_encoder.joblib` - Maps class indices to gesture names
                - `scaler.joblib` - Feature scaler (SVM only)
                - `metadata.json` - Model configuration and metrics
                
                **To use in gesture_recognition.py:**
                ```python
                import joblib
                
                # Load model
                model = joblib.load("models/{selected_model['model_name']}_{selected_model['timestamp']}/model.joblib")
                label_encoder = joblib.load("models/.../label_encoder.joblib")
                
                # Predict
                prediction = model.predict([normalized_landmarks])
                gesture_name = label_encoder.inverse_transform(prediction)[0]
                ```
                """)
                
                # Copy to standard location
                if st.button("📋 Set as Active Model", use_container_width=True):
                    import shutil
                    active_dir = MODELS_DIR / "active"
                    
                    # Remove old active
                    if active_dir.exists():
                        shutil.rmtree(active_dir)
                    
                    # Copy new active
                    shutil.copytree(selected_model['path'], active_dir)
                    st.success(f"✅ Set as active model: `models/active/`")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>🧠 <b>Gesture Model Training</b> | Phase 5</p>
        <p>Random Forest (fast) vs SVM OvO (accurate) | No neural networks needed!</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

