import time

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional
import base64
import fitz
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

class LineItem(BaseModel):
    description: str = Field(description="Description of the product or service")
    quantity: float = Field(default=1.0, description="Quantity purchased")
    unit_price: float = Field(default=0.0, description="Price per unit")
    line_total: float = Field(default=0.0, description="Total for this line item")

class InvoiceSchema(BaseModel):
    vendor_name: str = Field(description="Name of the vendor/company issuing the invoice")
    tax_id: Optional[str] = Field(description="Tax ID, VAT, or EIN of the vendor. Leave null if not found.")
    invoice_date: str = Field(description="Date the invoice was issued in YYYY-MM-DD format")
    due_date: Optional[str] = Field(description="Payment due date in YYYY-MM-DD format")
    subtotal: float = Field(description="Total amount before taxes")
    tax_amount: float = Field(description="Total tax amount applied")
    total_amount: float = Field(description="Final total amount due")
    line_items: List[LineItem] = Field(description="List of individual items on the invoice")


def process_uploaded_file(uploaded_file):
    """Converts uploaded Streamlit files into a list of base64 encoded strings."""
    base64_images = []

    if uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_bytes = uploaded_file.getvalue()
        base64_images.append(base64.b64encode(file_bytes).decode("utf-8"))

    elif uploaded_file.name.lower().endswith('.pdf'):
        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("jpeg")
            base64_images.append(base64.b64encode(img_bytes).decode("utf-8"))

    return base64_images


def extract_invoice_data(base64_images):
    """Passes images to Gemini and returns structured JSON + Token Usage."""
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.0, google_api_key=api_key)

    structured_llm = llm.with_structured_output(InvoiceSchema, include_raw=True)

    content = [{"type": "text",
                "text": "You are an expert financial auditor. Extract all financial data from the provided invoice images. Return the data EXACTLY matching the requested structure. Calculate missing subtotals if necessary."}]

    for b64_img in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
        })

    message = HumanMessage(content=content)
    result = structured_llm.invoke([message])

    parsed_data = result["parsed"].model_dump() if result.get("parsed") else {}

    tokens = 0
    raw_msg = result.get("raw")
    if raw_msg:
        if hasattr(raw_msg, 'usage_metadata') and raw_msg.usage_metadata:
            tokens = raw_msg.usage_metadata.get("total_tokens", 0)
        elif hasattr(raw_msg, 'response_metadata') and "token_usage" in raw_msg.response_metadata:
            tokens = raw_msg.response_metadata["token_usage"].get("totalTokens", 0)

    return parsed_data, tokens

st.set_page_config(page_title="Project Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .stAppDeployButton, 
    [data-testid="stAppDeployButton"], 
    [data-testid="stToolbar"] > div:first-child {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0px !important;
        height: 0px !important;
        pointer-events: none !important;
    }

    header[data-testid="stHeader"], 
    [data-testid="stHeaderActions"], 
    [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0px !important;
        z-index: -9999 !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Data Upload")
    uploaded_files = st.file_uploader("Upload Document", type=["jpg", "png", "pdf"], accept_multiple_files=True, max_upload_size=25)
    if st.button("Upload", use_container_width=True):
        if uploaded_files is not None:
            st.success("File uploaded successfully!")
        else:
            st.error("Please select a file first.")

st.title("InvoiceBot")
st.markdown("### *Intelligent Invoice and Contract Processing.*")
st.divider()

tab1, tab2, tab3 = st.tabs(["Executive Summary", "Data Visualizations", "ROI Analysis"])

with tab1:
    st.markdown("""
        ## Transforming Financial Operations with Intelligent Automation
        
        Welcome to the **InvoiceBot Living Lab**. This interactive demonstration showcases how modern AI can eliminate one of the most persistent bottlenecks in corporate finance: **manual document processing**.
        
        ### The Challenge: The Manual Data Entry Bottleneck
        Every month, Accounts Payable (AP) and procurement teams spend thousands of hours manually transcribing data from PDFs, scanned documents, and smartphone photos into central ERP systems. This traditional approach creates systemic issues:
        * **High Operational Costs:** Highly paid administrative professionals waste time on repetitive data entry instead of strategic financial analysis.
        * **Human Error:** Manual transcription naturally leads to costly typographical errors, miscalculated taxes, or incorrect vendor payments.
        * **Template Brittleness:** Older OCR (Optical Character Recognition) software fails when vendors change their invoice layouts because it relies on rigid, hard-coded templates.
        
        ### The Solution: Vision-Language Models (VLMs)
        **InvoiceBot** abandons outdated template-matching. Instead, it utilizes Google's **Gemini Vision AI**, which "reads" a document exactly like a human does. It understands spatial relationships, context, and complex nested tables regardless of how messy or unique the vendor's invoice format is.
        
        **Core Capabilities:**
        1. **Format Agnostic:** Drop in digital PDFs, scanned JPEGs, or PNGs. The AI visualizes them all instantly.
        2. **Structured Certainty:** The AI is strictly constrained to output exact, pre-defined financial fields (Vendor Name, Tax IDs, Subtotals, and individual Line Items). It will never return unstructured conversational text.
        3. **Human-in-the-Loop:** AI doesn't replace the human; it supercharges them. Extracted data is presented in a clean grid for rapid human validation before exporting to your accounting software.
        
        ### How to Use This Dashboard
        This application is designed to let you prove the business value using your own data in real-time:
        
        * **Step 1: Ingestion (Sidebar):** Upload a batch of sample invoices or contracts using the left-hand menu. 
        * **Step 2: Extraction & Validation (Tab 2):** Click "Start Document Processing" to watch the AI read the batch concurrently. Review the AI's work in the interactive data grid and export the clean, structured data directly to CSV or JSON.
        * **Step 3: ROI & Impact Analysis (Tab 3):** We don't just promise efficiency; we measure it. Navigate to the ROI tab to compare the exact fractional cent cost of the API and the seconds of validation time against your current manual labor baseline. 
        """)

with tab2:
    st.subheader("Data Visualization")

    if "extracted_data" in st.session_state and st.session_state.extracted_data:

        col1, col2 = st.columns([1, 2.5])

        with col1:
            st.subheader("Source Documents")
            st.write("Processed Files:")
            for doc in st.session_state.extracted_data:
                filename = doc.get("source_file", "Unknown File")
                st.caption(f"`{filename}`")

            if st.button("Reset / Clear Data"):
                del st.session_state.extracted_data
                st.session_state.pop("processing_time", None)
                st.session_state.pop("total_tokens", None)
                st.rerun()

        with col2:
            st.subheader("Extracted Data Grid")

            if "processing_time" in st.session_state and "total_tokens" in st.session_state:
                total_time = st.session_state.processing_time
                total_tokens = st.session_state.total_tokens
                doc_count = len(st.session_state.extracted_data)

                perf_col1, perf_col2, perf_col3 = st.columns(3)
                perf_col1.metric("Total Execution Time", f"{total_time:.2f} s")
                perf_col2.metric("Tokens Consumed", f"{total_tokens:,}")
                perf_col3.metric("Avg Time / Document", f"{(total_time / doc_count):.2f} s")
                st.markdown("<br>", unsafe_allow_html=True)  # visual spacer
            # ------------------------------------------

            raw_data = st.session_state.extracted_data
            df = pd.json_normalize(raw_data)

            if 'line_items' in df.columns:
                df['line_items'] = df['line_items'].astype(str)

            cols = ['source_file'] + [c for c in df.columns if c != 'source_file']
            df = df[cols]

            edited_df = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                disabled=True,
                num_rows="dynamic"
            )

            st.markdown("---")
            st.subheader("Export Verified Data")

            col_csv, col_json, col_empty = st.columns([1, 1, 2])
            with col_csv:
                csv_data = edited_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", data=csv_data, file_name="verified_invoices.csv", mime="text/csv",
                                   use_container_width=True)
            with col_json:
                json_data = edited_df.to_json(orient="records")
                st.download_button("Download JSON", data=json_data, file_name="verified_invoices.json",
                                   mime="application/json", use_container_width=True)

    elif uploaded_files:
        st.info(f"{len(uploaded_files)} document(s) queued for extraction.")

        if st.button("Start Document Processing", type="primary"):

            all_extracted_data = []
            total_tokens_used = 0

            start_time = time.time()

            with st.spinner(f"Extracting financial data from {len(uploaded_files)} document(s)..."):
                for file in uploaded_files:
                    b64_images = process_uploaded_file(file)

                    extracted_json, tokens = extract_invoice_data(b64_images)

                    extracted_json["source_file"] = file.name
                    all_extracted_data.append(extracted_json)
                    total_tokens_used += tokens

            end_time = time.time()
            st.session_state.extracted_data = all_extracted_data
            st.session_state.processing_time = end_time - start_time
            st.session_state.total_tokens = total_tokens_used

            st.rerun()

    else:
        st.warning("No documents found. Please upload PDF or JPG invoices in the sidebar to begin.")

with tab3:
        st.header("Living Lab: ROI & Impact Analysis")
        st.markdown("Calculate the operational savings by comparing manual data entry against AI-assisted processing.")

        st.subheader("Baseline Metrics")

        # Improved layout: Time input on the left, Wage input on the right
        col1, col2 = st.columns(2)

        with col1:
            time_input_col, time_label_col = st.columns([3, 1])
            with time_input_col:
                human_time_10 = st.number_input("Time to manually process 10 documents", min_value=1.0, value=45.0, step=1.0)
            with time_label_col:
                st.markdown("<div style='margin-top: 32px; font-weight: bold; color: gray;'>minutes</div>", unsafe_allow_html=True)

        with col2:
            wage_input_col, wage_label_col = st.columns([3, 1])
            with wage_input_col:
                monthly_wage = st.number_input("Employee's monthly wage", min_value=100.0, value=3500.0, step=100.0)
            with wage_label_col:
                st.markdown("<div style='margin-top: 32px; font-weight: bold; color: gray;'>USD / month</div>", unsafe_allow_html=True)

        hourly_rate = monthly_wage / 160
        per_minute_rate = hourly_rate / 60
        human_financial_cost = human_time_10 * per_minute_rate
        if "processing_time" in st.session_state and st.session_state.extracted_data:
            actual_time_sec = st.session_state.processing_time
            total_tokens = st.session_state.get("total_tokens", 0)
            docs_processed = len(st.session_state.extracted_data)

            avg_time_per_doc_min = (actual_time_sec / docs_processed) / 60
            model_time_10 = avg_time_per_doc_min * 10
            
            avg_tokens_per_doc = total_tokens / docs_processed
            tokens_for_10 = avg_tokens_per_doc * 10
        else:
            model_time_10 = 0.0
            tokens_for_10 = 0.0

        if "extracted_data" in st.session_state and st.session_state.extracted_data:
            total_tokens = st.session_state.get("total_tokens", 0)
            docs_processed = len(st.session_state.extracted_data)
            avg_tokens = total_tokens / docs_processed
            token_price = 0.15 / 1_000_000
            model_financial_cost = avg_tokens * token_price * 10

        else:
            model_financial_cost = 0.0

        st.markdown("---")
        st.subheader("Operational Cost (Labor for 10 Documents)")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.info("**Human Employee Processing**")
            st.metric(label="Total Time Cost", value=f"{human_time_10:.1f} mins")
            st.metric(label="Financial Cost (Labor)", value=f"${human_financial_cost:.2f}")

        with res_col2:
            st.success("**AI-Assisted Processing (Validation)**")
            if model_time_10 == 0.0:
                st.caption("Process documents in Tab 2 to see live AI metrics.")

            st.metric(
                label="Total Time Cost", 
                value=f"{model_time_10:.1f} mins", 
                delta=f"-{human_time_10 - model_time_10:.1f} mins", 
                delta_color="inverse"
            )
            st.metric(
                label="Financial Cost (Labor)", 
                value=f"${model_financial_cost:.4f}", 
                delta=f"-${human_financial_cost - model_financial_cost:.4f}", 
                delta_color="inverse"
            )

        st.markdown("---")
        st.subheader("API Token Cost Comparison (Per 10 Documents)")
        
        if tokens_for_10 > 0:
            st.write(f"**Estimated Token Usage for 10 documents:** `{tokens_for_10:,.0f} tokens`")
            
            pricing_data = {
                "Model Name": [
                    "Gemini 3.6 Flash (Current)", 
                    "GPT-4o-mini", 
                    "Claude 3 Haiku", 
                    "Claude 3.5 Sonnet", 
                    "GPT-4o"
                ],
                "Price per 1M Tokens (Blended)": [
                    0.15,  
                    0.20,  
                    0.35,  
                    4.00,  
                    6.00   
                ]
            }
            
            df_pricing = pd.DataFrame(pricing_data)
            df_pricing["Cost for 10 Documents"] = (tokens_for_10 / 1_000_000) * df_pricing["Price per 1M Tokens (Blended)"]
            
            df_pricing["Price per 1M Tokens (Blended)"] = df_pricing["Price per 1M Tokens (Blended)"].apply(lambda x: f"${x:.2f}")
            df_pricing["Cost for 10 Documents"] = df_pricing["Cost for 10 Documents"].apply(lambda x: f"${x:.6f}")

            st.dataframe(df_pricing, hide_index=True, use_container_width=True)
            
            st.caption("*Note: Vision-language API costs are highly dependent on image resolution and output length. Prices shown are estimated blended averages of input/output rates. The True Operational Cost is Validation Labor + API Cost.*")
        else:
            st.info("Process documents in Tab 2 to calculate live API token costs across top industry models.")