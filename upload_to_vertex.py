"""
Upload documents (PDFs) to GCS and import to Vertex AI Search Datastore.
"""

import logging
from pathlib import Path
from google.cloud import storage
from google.cloud import discoveryengine_v1 as discoveryengine

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def check_upload_config():
    """Validate configuration for upload."""
    errors = []
    
    if not config.PROJECT_ID:
        errors.append("GOOGLE_CLOUD_PROJECT environment variable is required")
    
    if not config.DATASTORE_ID:
        errors.append("VERTEX_AI_DATASTORE_ID environment variable is required")
    
    if not config.GCS_BUCKET:
        errors.append("GCS_BUCKET environment variable is required")
    
    return errors


def upload_pdfs_to_gcs(pdf_dir: str = "./documents/pdfs") -> tuple[str, int]:
    """
    Upload PDFs to Google Cloud Storage.
    
    Returns: (gcs_uri, count of uploaded files)
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")
    
    pdfs = list(pdf_path.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDFs found in {pdf_dir}")
    
    gcs_bucket = config.GCS_BUCKET
    gcs_folder = config.GCS_FOLDER
    
    logger.info(f"Uploading {len(pdfs)} PDFs to gs://{gcs_bucket}/{gcs_folder}/")
    
    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(gcs_bucket)
    
    uploaded = 0
    for pdf in pdfs:
        blob_name = f"{gcs_folder}/{pdf.name}"
        blob = bucket.blob(blob_name)
        
        if blob.exists():
            logger.info(f"  Exists: {pdf.name}")
        else:
            blob.upload_from_filename(str(pdf))
            logger.info(f"  Uploaded: {pdf.name}")
            uploaded += 1
    
    gcs_uri = f"gs://{gcs_bucket}/{gcs_folder}/"
    logger.info(f"✅ Done! {uploaded} new files uploaded to {gcs_uri}")
    return gcs_uri, uploaded


def import_to_datastore(gcs_uri: str) -> str:
    """
    Import documents from GCS to Vertex AI Search datastore.
    
    Returns: Operation name for tracking
    """
    client = discoveryengine.DocumentServiceClient()
    
    parent = (
        f"projects/{config.PROJECT_ID}"
        f"/locations/global"
        f"/collections/default_collection"
        f"/dataStores/{config.DATASTORE_ID}"
        f"/branches/default_branch"
    )
    
    # Import PDFs from GCS
    request = discoveryengine.ImportDocumentsRequest(
        parent=parent,
        gcs_source=discoveryengine.GcsSource(
            input_uris=[f"{gcs_uri.rstrip('/')}/*.pdf"],
            data_schema="content"  # For unstructured docs like PDFs
        ),
        reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL
    )
    
    logger.info(f"Starting import from {gcs_uri}...")
    operation = client.import_documents(request=request)
    
    logger.info(f"✅ Import started!")
    logger.info(f"   Operation: {operation.operation.name}")
    
    return operation.operation.name


def check_status(operation_name: str) -> bool:
    """Check import operation status."""
    from google.longrunning import operations_pb2
    
    client = discoveryengine.DocumentServiceClient()
    request = operations_pb2.GetOperationRequest(name=operation_name)
    op = client._transport.operations_client.get_operation(request)
    
    if op.done:
        if op.error.code:
            logger.error(f"❌ Import failed: {op.error.message}")
        else:
            logger.info("✅ Import complete!")
        return True
    else:
        logger.info("⏳ Import still in progress...")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload documents to Vertex AI Search")
    parser.add_argument('--pdf-dir', '-d', default='./documents/pdfs', help='PDF directory')
    parser.add_argument('--skip-upload', action='store_true', help='Skip GCS upload, just import')
    parser.add_argument('--check', help='Check status of operation')
    
    args = parser.parse_args()
    
    if args.check:
        check_status(args.check)
        return
    
    # Validate configuration
    errors = check_upload_config()
    if errors:
        print("❌ Configuration errors:")
        for e in errors:
            print(f"   - {e}")
        print("\nPlease set the required environment variables. See .env.example")
        return
    
    gcs_bucket = config.GCS_BUCKET
    gcs_folder = config.GCS_FOLDER
    
    # Step 1: Upload to GCS
    if not args.skip_upload:
        gcs_uri, count = upload_pdfs_to_gcs(args.pdf_dir)
        if count == 0:
            logger.info("No new files to upload. Use --skip-upload to just trigger import.")
    else:
        gcs_uri = f"gs://{gcs_bucket}/{gcs_folder}/"
    
    # Step 2: Import to datastore
    print("\n" + "="*60)
    operation = import_to_datastore(gcs_uri)
    
    print("\n" + "="*60)
    print(f"📄 PDFs location: {gcs_uri}")
    print(f"🔄 Import operation: {operation}")
    print(f"\nCheck status with:")
    print(f"  python upload_to_vertex.py --check '{operation}'")
    print("="*60)


if __name__ == "__main__":
    main()
