import streamlit as st


@st.cache_data
def transform(df, selected_company, selected_industries, selected_skills):
    filtered_df = df.copy()
    # Apply filters
    if selected_company != "All":
        filtered_df = filtered_df[filtered_df["company_name"] == selected_company]
    if selected_industries:
        filtered_df = filtered_df[
            filtered_df["industries"].apply(
                lambda x: any(industry in x for industry in selected_industries)
            )
        ]
    if selected_skills:
        filtered_df = filtered_df[
            filtered_df["skills"].apply(
                lambda x: any(skill in x for skill in selected_skills)
            )
        ]
    return filtered_df


if "config_file" not in st.session_state:
    st.write("Load data from Load Data page")


else:
    st.write("Data loaded")
    st.write(
        st.session_state["kaggle_data"].shape,
        st.session_state["kaggle_data_embeddings"].shape,
        st.session_state["skills_s"].shape,
        st.session_state["industries_s"].shape,
    )

    industries_s = st.session_state["industries_s"]
    skills_s = st.session_state["skills_s"]
    # --- Filters ---
    st.sidebar.header("Filters")

    # Company Name filter ()
    company_names = sorted(st.session_state["kaggle_data"]["company_name"].unique())
    selected_company = st.sidebar.selectbox(
        "Company Name", ["All"] + list(company_names)
    )

    # Industry filter (multi-selection)
    selected_industries = st.sidebar.multiselect("Industry", industries_s.tolist())

    # Skills filter (multi-selection)
    selected_skills = st.sidebar.multiselect("Skills", skills_s.tolist())

    # --- Filter data ---
    if st.button("Filter data"):
        filtered_df = transform(
            st.session_state["kaggle_data"],
            selected_company,
            selected_industries,
            selected_skills,
        )
        # --- Display filtered data ---
        st.header("Filtered Job Postings")
        st.dataframe(
            filtered_df[
                [
                    "company_name",
                    "title",
                    "description",
                    "skills",
                    "industries",
                    "med_salary_monthly",
                ]
            ].reset_index(drop=True),
            height=500,
        )
        st.write("Data filtered successfully!")

        # Save in session state the filtered data to be used in other pages
        if "filtered_data" not in st.session_state["output_data"]:
            st.session_state["output_data"]["filtered_data"] = []
        parameters = (selected_company, selected_industries, selected_skills)
        st.session_state["output_data"]["filtered_data"].append(
            (parameters, filtered_df)
        )
        # Save the last parameters used to filter the data
        if "last_filter_parameters" not in st.session_state:
            st.session_state["last_filter_parameters"] = None
        st.session_state["last_filter_parameters"] = parameters

        st.write(f"Filter parameters: {st.session_state["last_filter_parameters"]}")
