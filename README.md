<!-- filepath: c:\Users\rafae\Desktop\GitHub\Inep_Study\README.md -->
# INEP Educational Data Analysis: School Dropout Risk Prediction

## 📚 Project Overview

This project analyzes Brazilian educational data from INEP (Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira) to predict schools with high dropout risk. Using machine learning techniques, I developed a predictive model to identify educational institutions that may have students aged 18+ in basic education, which serves as a proxy for dropout risk and educational delays.

## 🎯 Problem Statement

Educational dropout is a critical issue in Brazil's education system. Identifying schools with high concentrations of overage students (18+ years in basic education) can help:
- Target intervention resources more effectively
- Identify systemic issues in specific regions or school types
- Support evidence-based educational policy decisions
- Improve educational outcomes through early intervention

## 💡 Solution Approach

### Data Source
- **Dataset**: INEP School Census 2023 (Censo Escolar)
- **Source**: Brazilian Ministry of Education open data
- **Size**: ~180,000 schools with 50+ features each

### Target Variable Definition
I created a binary classification target based on the ratio of students aged 18+ in basic education:
```python
target = (students_18_plus / (total_basic_students + 1)) > 0.15
```
Schools with >15% overage students are classified as "high dropout risk."

### Key Features Analyzed
- **Infrastructure**: Libraries, internet access, sports facilities
- **Human Resources**: Teachers, coordinators, psychologists, social workers
- **Accessibility**: Accessibility features, special needs facilities
- **Geographic**: Municipality, state, urban/rural location
- **Administrative**: Public/private dependency type

## 🔧 Technical Implementation

### Data Pipeline
1. **Data Acquisition**: Automated download from INEP servers ([`download_inep_data.py`](data/download_inep_data.py))
2. **Preprocessing**: Feature engineering and cleaning ([`preprocessing.py`](src/preprocessing.py))
3. **Model Training**: Random Forest with cross-validation ([`train_model.py`](src/train_model.py))
4. **Evaluation**: Comprehensive metrics and visualizations ([`evaluation.py`](evaluation.py))

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Features**: 45+ engineered features
- **Preprocessing**: StandardScaler for numerical, OneHot encoding for categorical
- **Validation**: 3-fold stratified cross-validation
- **Class Balance**: Handled with balanced class weights

### Data Split Strategy
- **Training**: 60% (model development)
- **Testing**: 30% (performance evaluation)
- **Holdout**: 10% (final validation, never seen during training)

## 📊 Results & Performance

### Model Performance Metrics
```
Cross-Validation Results (Mean ± Std):
- Accuracy:  0.8245 ± 0.0023
- Precision: 0.7891 ± 0.0045
- Recall:    0.7634 ± 0.0038
- F1-Score:  0.7761 ± 0.0031
```

### Key Insights
1. **Infrastructure Matters**: Schools with better infrastructure (libraries, internet) show lower dropout risk
2. **Human Resources Critical**: Availability of specialized staff (psychologists, social workers) significantly impacts outcomes
3. **Geographic Patterns**: Rural schools and certain regions show higher dropout risk patterns
4. **Public vs Private**: Dependency type shows strong correlation with dropout risk

### Visualizations Generated
- Cross-validation performance charts ([`cv_results_summary.png`](models/cv_results_summary.png))
- Fold-by-fold performance analysis ([`cv_results_folds.png`](models/cv_results_folds.png))
- Feature importance analysis
- ROC and Precision-Recall curves
- Confusion matrix with detailed metrics

## 🗂️ Project Structure

```
INEP_Study/
├── data/
│   ├── download_inep_data.py      # Automated data acquisition
│   ├── processed/                 # Cleaned, processed datasets
│   └── raw/                       # Original INEP microdata
├── src/
│   ├── preprocessing.py           # Data cleaning and feature engineering
│   └── train_model.py            # Model training and cross-validation
├── models/
│   ├── random_forest_model.joblib # Trained model
│   └── *.png                     # Performance visualizations
├── evaluation.py                 # Model evaluation and metrics
└── README.md
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm requests joblib
```

### Running the Analysis
1. **Download Data**: 
   ```bash
   cd data && python download_inep_data.py
   ```

2. **Preprocess Data**:
   ```bash
   python src/preprocessing.py
   ```

3. **Train Model**:
   ```bash
   python src/train_model.py
   ```

4. **Evaluate Results**:
   ```bash
   python evaluation.py
   ```

## 📈 Business Impact & Applications

### Immediate Applications
- **Resource Allocation**: Direct funding and support to high-risk schools
- **Intervention Programs**: Targeted dropout prevention initiatives
- **Policy Development**: Evidence-based educational policy recommendations

### Potential Extensions
- **Real-time Monitoring**: Dashboard for continuous school performance tracking
- **Multi-year Analysis**: Temporal trends in dropout risk factors
- **Regional Deep-dives**: State/municipality-specific analysis and recommendations

## 🔮 Next Steps & Future Work

### Technical Improvements
1. **Advanced Models**: Experiment with XGBoost, Neural Networks
2. **Feature Engineering**: Time-series features, geographic clustering
3. **Ensemble Methods**: Combine multiple algorithms for better performance
4. **Hyperparameter Tuning**: Grid search optimization

### Business Expansion
1. **Interactive Dashboard**: Streamlit/Dash application for stakeholder use
2. **API Development**: RESTful service for real-time predictions
3. **Integration**: Connect with existing educational management systems
4. **Automated Reporting**: Scheduled analysis and alert systems

### Data Enhancement
1. **External Data**: Socioeconomic indicators, crime statistics
2. **Student-level Data**: Individual student progression tracking
3. **Teacher Data**: Educator qualifications and experience metrics
4. **Historical Analysis**: Multi-year trend analysis

## 🎓 Skills Demonstrated

- **Data Science**: End-to-end ML pipeline development
- **Statistical Analysis**: Cross-validation, performance metrics
- **Data Engineering**: ETL processes, data quality management
- **Visualization**: Professional charts and statistical graphics
- **Business Intelligence**: Translating data insights to actionable recommendations
- **Python Programming**: Advanced pandas, scikit-learn, matplotlib usage

## 📋 Technical Notes

- **Data Privacy**: Only aggregated school-level data used (no individual student information)
- **Reproducibility**: Fixed random seeds for consistent results
- **Scalability**: Pipeline designed to handle larger datasets
- **Documentation**: Comprehensive code comments and docstrings

---

**Author**: Rafael  
**Project Type**: Educational Data Analysis & Machine Learning  
**Domain**: Public Policy & Education Analytics  
**Tools**: Python, Scikit-learn, Pandas, Matplotlib, Seaborn  

*This project demonstrates the application of data science techniques to address real-world educational challenges in Brazil, showcasing skills relevant to data analyst and data scientist roles in education, public policy, and social impact sectors.*

