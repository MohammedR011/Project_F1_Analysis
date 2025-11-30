import streamlit as st
import os
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Page configuration
st.set_page_config(page_title="Formula 1 Analysis", page_icon="🏎️", layout="wide")

# Title
st.title("🏎️ Formula 1 World Championship Analysis (1950-2024)")
st.markdown("---")

# Sidebar
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", [
    "📊 Data Overview",
    "🏆 Top Winners",
    "📈 Races per Season",
    "🏁 Top Circuits",
    "🔄 Qualifying vs Race Position",
    "🏢 Top Constructors",
    "🌍 Driver Nationalities",
    "🤖 ML Model Training",
    "🎯 Win Prediction"
])

# Cache data loading
@st.cache_data
def load_data():
    with st.spinner("Downloading and loading dataset..."):
        path = kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")
        
        drivers_path = os.path.join(path, "drivers.csv")
        races_path = os.path.join(path, "races.csv")
        results_path = os.path.join(path, "results.csv")
        qualifying_path = os.path.join(path, "qualifying.csv")
        constructors_path = os.path.join(path, "constructors.csv")
        
        df_drivers = pd.read_csv(drivers_path)
        df_races = pd.read_csv(races_path)
        df_results = pd.read_csv(results_path)
        df_qualifying = pd.read_csv(qualifying_path)
        df_constructors = pd.read_csv(constructors_path)
        
        # Clean data
        df_drivers.dropna(inplace=True)
        df_races.dropna(inplace=True)
        df_results.dropna(inplace=True)
        df_qualifying.dropna(inplace=True)
        
        return df_drivers, df_races, df_results, df_qualifying, df_constructors, path

# Load data
df_drivers, df_races, df_results, df_qualifying, df_constructors, path = load_data()

# Section: Data Overview
if section == "📊 Data Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Drivers", df_drivers.shape[0])
    col2.metric("Races", df_races.shape[0])
    col3.metric("Results", df_results.shape[0])
    col4.metric("Qualifying Records", df_qualifying.shape[0])
    
    st.subheader("Dataset Shapes")
    st.write(f"- **Drivers:** {df_drivers.shape}")
    st.write(f"- **Races:** {df_races.shape}")
    st.write(f"- **Results:** {df_results.shape}")
    st.write(f"- **Qualifying:** {df_qualifying.shape}")
    
    st.subheader("Sample Data")
    data_choice = st.selectbox("Select dataset to preview:", ["Drivers", "Races", "Results", "Qualifying"])
    
    if data_choice == "Drivers":
        st.dataframe(df_drivers.head(10))
    elif data_choice == "Races":
        st.dataframe(df_races.head(10))
    elif data_choice == "Results":
        st.dataframe(df_results.head(10))
    elif data_choice == "Qualifying":
        st.dataframe(df_qualifying.head(10))

# Section: Top Winners
elif section == "🏆 Top Winners":
    st.header("🏆 Top 10 Winning Drivers")
    
    winners = df_results[df_results['positionOrder'] == 1]
    winners = winners.merge(df_drivers, left_on='driverId', right_on='driverId')
    winners['driver_name'] = winners['forename'] + ' ' + winners['surname']
    top_winners = winners['driver_name'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    top_winners.plot(kind='bar', color='red', ax=ax)
    ax.set_title("Top 10 Winning Drivers", fontsize=16)
    ax.set_xlabel("Driver", fontsize=12)
    ax.set_ylabel("Number of Wins", fontsize=12)
    plt.xticks(rotation=50)
    st.pyplot(fig)
    
    st.subheader("Win Statistics")
    st.dataframe(top_winners.reset_index().rename(columns={'index': 'Driver', 'driver_name': 'Wins'}))

# Section: Races per Season
elif section == "📈 Races per Season":
    st.header("📈 Number of Races per Season")
    
    races_per_year = df_races['year'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    races_per_year.plot(kind='line', marker='o', ax=ax)
    ax.set_title("Number of Races per Season", fontsize=16)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Races", fontsize=12)
    ax.grid(True)
    st.pyplot(fig)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Races/Season", f"{races_per_year.mean():.1f}")
    col2.metric("Max Races (Season)", races_per_year.max())
    col3.metric("Min Races (Season)", races_per_year.min())

# Section: Top Circuits
elif section == "🏁 Top Circuits":
    st.header("🏁 Top 10 Most Common Race Locations")
    
    top_circuits = df_races['name'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    top_circuits.plot(kind='bar', ax=ax)
    ax.set_title("Top 10 Most Common Race Locations", fontsize=16)
    ax.set_xlabel("Circuit Name", fontsize=12)
    ax.set_ylabel("Number of Races Hosted", fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("Circuit Statistics")
    st.dataframe(top_circuits.reset_index().rename(columns={'index': 'Circuit', 'name': 'Races Hosted'}))

# Section: Qualifying vs Race Position
elif section == "🔄 Qualifying vs Race Position":
    st.header("🔄 Qualifying Position vs Final Race Position")
    
    df_qual_race = pd.merge(
        df_qualifying[['raceId', 'driverId', 'position']],
        df_results[['raceId', 'driverId', 'positionOrder']],
        on=['raceId', 'driverId'],
        how='inner'
    )
    
    df_qual_race.rename(columns={'position': 'qualifying_position', 'positionOrder': 'race_position'}, inplace=True)
    df_qual_race['qualifying_position'] = pd.to_numeric(df_qual_race['qualifying_position'], errors='coerce')
    df_qual_race['race_position'] = pd.to_numeric(df_qual_race['race_position'], errors='coerce')
    df_qual_race.dropna(inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(data=df_qual_race, x='qualifying_position', y='race_position', 
                scatter_kws={'alpha': 0.3}, line_kws={"color": "red"}, ax=ax)
    ax.set_title('Qualifying Position vs Final Race Position', fontsize=16)
    ax.set_xlabel('Qualifying Position (Lower = Better)', fontsize=12)
    ax.set_ylabel('Race Position (Lower = Better)', fontsize=12)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True)
    st.pyplot(fig)
    
    correlation = df_qual_race['qualifying_position'].corr(df_qual_race['race_position'])
    st.metric("Correlation Coefficient", f"{correlation:.2f}")
    st.info("A strong positive correlation indicates that qualifying position significantly affects race outcome.")

# Section: Top Constructors
elif section == "🏢 Top Constructors":
    st.header("🏢 Top 10 Constructors by Total Wins")
    
    winners = df_results[df_results['positionOrder'] == 1]
    winners = winners.merge(df_constructors, on='constructorId')
    team_wins = winners['name'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    team_wins.plot(kind='bar', color='steelblue', ax=ax)
    ax.set_title("Top 10 Constructors by Total Wins", fontsize=16)
    ax.set_xlabel("Constructor", fontsize=12)
    ax.set_ylabel("Number of Wins", fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("Constructor Win Statistics")
    st.dataframe(team_wins.reset_index().rename(columns={'index': 'Constructor', 'name': 'Wins'}))

# Section: Driver Nationalities
elif section == "🌍 Driver Nationalities":
    st.header("🌍 Top 10 Driver Nationalities in F1 History")
    
    nationalities = df_drivers['nationality'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    nationalities.plot(kind='bar', color='red', ax=ax)
    ax.set_title("Top 10 Driver Nationalities in F1 History", fontsize=16)
    ax.set_ylabel("Number of Drivers", fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("Nationality Statistics")
    st.dataframe(nationalities.reset_index().rename(columns={'index': 'Nationality', 'nationality': 'Number of Drivers'}))

# Section: ML Model Training
elif section == "🤖 ML Model Training":
    st.header("🤖 Machine Learning Model Training")
    
    with st.spinner("Preparing features and training models..."):
        # Prepare features
        races_data = df_races[['raceId', 'year', 'round', 'name', 'circuitId']]
        drivers_data = df_drivers[['driverId', 'forename', 'surname', 'nationality']]
        qualifying_data = df_qualifying[['raceId', 'driverId', 'position']]
        results_data = df_results[['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder', 'points']]
        
        qualifying_data = qualifying_data.rename(columns={'position': 'qualifying_position'})
        results_data = results_data.rename(columns={'positionOrder': 'race_position'})
        
        df_features = pd.merge(results_data, races_data, on='raceId', how='left')
        df_features = pd.merge(df_features, drivers_data, on='driverId', how='left')
        df_features = pd.merge(df_features, qualifying_data, on=['raceId', 'driverId'], how='left')
        
        df_features['qualifying_position'] = df_features['qualifying_position'].fillna(30)
        df_features['qualifying_position'] = pd.to_numeric(df_features['qualifying_position'], errors='coerce')
        df_features['winner'] = df_features['race_position'].apply(lambda x: 1 if x == 1 else 0)
        df_features.dropna(subset=['race_position'], inplace=True)
        
        X = df_features[['qualifying_position', 'grid', 'year', 'round', 'constructorId']]
        y = df_features['winner']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train models
        log_reg = LogisticRegression(class_weight='balanced', max_iter=1000)
        log_reg.fit(X_train, y_train)
        
        rf_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=200)
        rf_model.fit(X_train, y_train)
        
        scale_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])
        xgb_model = XGBClassifier(scale_pos_weight=scale_weight, eval_metric='auc', random_state=42, n_estimators=300)
        xgb_model.fit(X_train, y_train)
        
        # Store in session state
        st.session_state['models_trained'] = True
        st.session_state['log_reg'] = log_reg
        st.session_state['rf_model'] = rf_model
        st.session_state['xgb_model'] = xgb_model
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
    
    st.success("✅ Models trained successfully!")
    
    # Model comparison
    models = {
        "Logistic Regression": log_reg,
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }
    
    results_data = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results_data.append({
            'Model': name,
            'Accuracy': f"{acc:.2f}",
            'F1 Score': f"{f1:.2f}",
            'ROC-AUC': f"{roc_auc:.2f}"
        })
    
    st.subheader("Model Performance Comparison")
    st.dataframe(pd.DataFrame(results_data))
    
    # Confusion matrices
    st.subheader("Confusion Matrices")
    
    # Add explanation
    with st.expander("ℹ️ What is a Confusion Matrix?"):
        st.markdown("""
        A **Confusion Matrix** is a table that helps us understand how well our model predicts race winners.
        
        **Matrix Layout:**
        - **True Negatives (TN)** - Top Left: Correctly predicted as NOT winner
        - **False Positives (FP)** - Top Right: Incorrectly predicted as winner (but didn't win)
        - **False Negatives (FN)** - Bottom Left: Incorrectly predicted as NOT winner (but actually won)
        - **True Positives (TP)** - Bottom Right: Correctly predicted as winner
        
        **Good Model Characteristics:**
        - High numbers on the diagonal (TN and TP) = correct predictions
        - Low numbers off the diagonal (FP and FN) = incorrect predictions
        
        **In F1 Context:**
        - **True Positive**: Model predicted the driver would win, and they did! 🏆
        - **True Negative**: Model predicted the driver wouldn't win, and they didn't ✓
        - **False Positive**: Model predicted win, but driver didn't win ❌
        - **False Negative**: Model didn't predict win, but driver won (missed opportunity) ❌
        """)
    
    cols = st.columns(3)
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate totals
        tn, fp, fn, tp = cm.ravel()
        total_winners = tp + fn  # Actual winners
        total_non_winners = tn + fp  # Actual non-winners
        
        with cols[idx]:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"{name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
            # Add statistics below each confusion matrix
            st.markdown(f"""
            **Model Statistics:**
            - ✅ Correct Predictions: **{tn + tp:,}**
            - ❌ Incorrect Predictions: **{fp + fn:,}**
            - 🏆 Total Actual Winners: **{total_winners:,}**
            - 📊 Total Actual Non-Winners: **{total_non_winners:,}**
            
            **Breakdown:**
            - True Positives (Correct Winner): **{tp}**
            - True Negatives (Correct Non-Winner): **{tn:,}**
            - False Positives (Wrong Winner): **{fp}**
            - False Negatives (Missed Winner): **{fn}**
            """)
    
    # Feature importance for XGBoost
    st.subheader("Feature Importance (XGBoost)")
    importance = xgb_model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values(by='Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
    ax.set_title("Top 10 Important Features for Predicting F1 Race Winner", fontsize=14)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

# Section: Win Prediction
elif section == "🎯 Win Prediction":
    st.header("🎯 Predict Driver Win Probability")
    
    if 'models_trained' not in st.session_state:
        st.warning("⚠️ Please train the models first by visiting the 'ML Model Training' section!")
    else:
        st.info("Select a driver from the dropdown to predict their win probability based on their most recent race data.")
        
        # Create driver list with full names
        df_drivers['full_name'] = df_drivers['forename'] + ' ' + df_drivers['surname']
        driver_list = sorted(df_drivers['full_name'].unique().tolist())
        
        # Add search functionality
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_driver = st.selectbox(
                "Select a Driver:",
                options=driver_list,
                index=driver_list.index('Lando Norris') if 'Lando Norris' in driver_list else 0
            )
        
        with col2:
            st.write("")
            st.write("")
            predict_button = st.button("🏁 Predict Win Probability", type="primary")
        
        # Optional: Show driver info before prediction
        if selected_driver:
            driver_info = df_drivers[df_drivers['full_name'] == selected_driver].iloc[0]
            
            with st.expander("ℹ️ Driver Information"):
                info_col1, info_col2, info_col3 = st.columns(3)
                info_col1.write(f"**Nationality:** {driver_info['nationality']}")
                info_col2.write(f"**Driver Number:** {driver_info['number'] if 'number' in driver_info and pd.notna(driver_info['number']) else 'N/A'}")
                info_col3.write(f"**DOB:** {driver_info['dob'] if 'dob' in driver_info else 'N/A'}")
        
        if predict_button and selected_driver:
            # Extract last name from selected driver
            last_name = selected_driver.split()[-1]
            
            with st.spinner(f"Analyzing {selected_driver}'s race data..."):
                matched_drivers = df_drivers[df_drivers['surname'].str.lower() == last_name.lower()]
                
                if matched_drivers.empty:
                    st.error(f"❌ No driver found with last name '{last_name}'.")
                else:
                    driver_results = (
                        df_results
                        .merge(df_races, on='raceId', how='left')
                        .merge(df_drivers, on='driverId', how='left')
                        .merge(df_qualifying, on=['raceId', 'driverId'], how='left')
                    )
                    
                    driver_data = driver_results[driver_results['surname'].str.lower() == last_name.lower()].sort_values(by='date', ascending=False).head(1)
                    
                    if driver_data.empty:
                        st.error(f"❌ No race data found for {selected_driver}.")
                    else:
                        X_test = st.session_state['X_test']
                        xgb_model = st.session_state['xgb_model']
                        
                        feature_cols = X_test.columns
                        available_features = [col for col in feature_cols if col in driver_data.columns]
                        
                        X_input = driver_data[available_features].copy()
                        
                        for col in feature_cols:
                            if col not in X_input.columns:
                                X_input.loc[:, col] = 0
                        
                        X_input = X_input[feature_cols]
                        
                        win_prob = xgb_model.predict_proba(X_input)[:, 1][0] * 100
                        
                        st.success("✅ Prediction Complete!")
                        st.markdown("---")
                        
                        # Display results in a more attractive way
                        st.subheader(f"🏎️ {selected_driver}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("📅 Race", f"{driver_data['name'].values[0][:30]}...")
                        col2.metric("📆 Year", f"{int(driver_data['year'].values[0])}")
                        col3.metric("🚦 Grid Position", f"P{int(driver_data['grid'].values[0])}")
                        col4.metric("🏁 Finish Position", f"P{int(driver_data['positionOrder'].values[0])}")
                        
                        st.markdown("---")
                        
                        # Win probability with visual indicator
                        st.subheader("Win Probability Prediction")
                        
                        # Create a progress bar for visual representation
                        st.progress(min(win_prob / 100, 1.0))
                        
                        # Large metric display
                        prob_col1, prob_col2, prob_col3 = st.columns([1, 2, 1])
                        with prob_col2:
                            st.markdown(f"<h1 style='text-align: center; color: #FF1801;'>{win_prob:.2f}%</h1>", unsafe_allow_html=True)
                        
                        # Interpretation
                        st.markdown("---")
                        if win_prob > 50:
                            st.success("🎉 **HIGH CHANCE OF WINNING!** This driver has excellent odds based on historical data.")
                        elif win_prob > 25:
                            st.info("💪 **MODERATE CHANCE OF WINNING** - The driver has a reasonable shot at victory.")
                        elif win_prob > 10:
                            st.warning("📊 **LOWER CHANCE OF WINNING** - Victory is possible but less likely.")
                        else:
                            st.error("📉 **LOW CHANCE OF WINNING** - Based on the data, winning is unlikely.")
                        
                        # Additional insights
                        with st.expander("📊 View Race Details"):
                            st.write(f"**Circuit:** {driver_data['name'].values[0]}")
                            st.write(f"**Starting Grid:** Position {int(driver_data['grid'].values[0])}")
                            if 'position' in driver_data.columns and pd.notna(driver_data['position'].values[0]):
                                st.write(f"**Qualifying Position:** {driver_data['position'].values[0]}")
                            else:
                                st.write(f"**Qualifying Position:** N/A")
                            if 'constructorId_x' in driver_data.columns:
                                st.write(f"**Constructor ID:** {int(driver_data['constructorId_x'].values[0])}")
                            elif 'constructorId_y' in driver_data.columns:
                                st.write(f"**Constructor ID:** {int(driver_data['constructorId_y'].values[0])}")
                            elif 'constructorId' in driver_data.columns:
                                st.write(f"**Constructor ID:** {int(driver_data['constructorId'].values[0])}")
                            if 'round' in driver_data.columns:
                                st.write(f"**Round:** {int(driver_data['round'].values[0])}")
        
        # Show recent predictions comparison
        st.markdown("---")
        st.subheader("🔍 Compare Multiple Drivers")
        
        multi_drivers = st.multiselect(
            "Select drivers to compare (max 5):",
            options=driver_list,
            max_selections=5
        )
        
        if st.button("Compare Drivers") and multi_drivers:
            comparison_data = []
            
            for driver_name in multi_drivers:
                last_name = driver_name.split()[-1]
                
                driver_results = (
                    df_results
                    .merge(df_races, on='raceId', how='left')
                    .merge(df_drivers, on='driverId', how='left')
                    .merge(df_qualifying, on=['raceId', 'driverId'], how='left')
                )
                
                driver_data = driver_results[driver_results['surname'].str.lower() == last_name.lower()].sort_values(by='date', ascending=False).head(1)
                
                if not driver_data.empty:
                    X_test = st.session_state['X_test']
                    xgb_model = st.session_state['xgb_model']
                    
                    feature_cols = X_test.columns
                    available_features = [col for col in feature_cols if col in driver_data.columns]
                    
                    X_input = driver_data[available_features].copy()
                    
                    for col in feature_cols:
                        if col not in X_input.columns:
                            X_input.loc[:, col] = 0
                    
                    X_input = X_input[feature_cols]
                    
                    win_prob = xgb_model.predict_proba(X_input)[:, 1][0] * 100
                    
                    comparison_data.append({
                        'Driver': driver_name,
                        'Win Probability': f"{win_prob:.2f}%",
                        'Grid Position': f"P{int(driver_data['grid'].values[0])}",
                        'Year': int(driver_data['year'].values[0])
                    })
            
            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Data Source:** Formula 1 World Championship (1950-2024) from Kaggle")