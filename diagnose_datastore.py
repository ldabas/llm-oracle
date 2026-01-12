#!/usr/bin/env python3
"""
Diagnostic tool to investigate Vertex AI Search datastore and retrieval
"""
from google.cloud import discoveryengine_v1 as discoveryengine

import config

# Use configuration from config module
PROJECT_ID = config.PROJECT_ID
LOCATION = "global"
DATASTORE_ID = config.DATASTORE_ID


def check_config():
    """Verify configuration is set."""
    print("\n" + "="*60)
    print("⚙️  CONFIGURATION CHECK")
    print("="*60)
    
    errors = config.validate_config()
    if errors:
        print("\n❌ Configuration errors:")
        for e in errors:
            print(f"   - {e}")
        print("\nPlease set the required environment variables and try again.")
        print("See .env.example for required variables.")
        return False
    
    print(f"\n✅ Configuration looks good!")
    print(f"   Project: {PROJECT_ID}")
    print(f"   Datastore: {DATASTORE_ID}")
    return True


def check_datastore_documents():
    """List documents in the datastore to verify what's indexed."""
    print("\n" + "="*60)
    print("🔍 DATASTORE DOCUMENT CHECK")
    print("="*60)
    
    try:
        # Create client
        client = discoveryengine.DocumentServiceClient()
        
        # The datastore branch path
        parent = f"projects/{PROJECT_ID}/locations/{LOCATION}/dataStores/{DATASTORE_ID}/branches/default_branch"
        
        print(f"\n📦 Datastore: {DATASTORE_ID}")
        print(f"📍 Parent: {parent}\n")
        
        # List documents
        request = discoveryengine.ListDocumentsRequest(
            parent=parent,
            page_size=100  # Get up to 100 per page
        )
        
        total_docs = 0
        doc_samples = []
        
        page_result = client.list_documents(request=request)
        
        for document in page_result:
            total_docs += 1
            if len(doc_samples) < 10:  # Collect first 10 as samples
                doc_samples.append({
                    "id": document.id,
                    "name": document.name.split("/")[-1] if "/" in document.name else document.name,
                })
        
        # Count additional pages
        while page_result.next_page_token:
            request = discoveryengine.ListDocumentsRequest(
                parent=parent,
                page_size=100,
                page_token=page_result.next_page_token
            )
            page_result = client.list_documents(request=request)
            for _ in page_result:
                total_docs += 1
        
        print(f"📊 Total documents indexed: {total_docs}")
        print(f"\n📄 Sample documents (first 10):")
        for i, doc in enumerate(doc_samples, 1):
            print(f"   {i}. {doc['id']}")
        
        if total_docs < 2000:
            print(f"\n⚠️  WARNING: Only {total_docs} documents indexed (expected ~2000)")
            print("   Possible causes:")
            print("   - Documents still being indexed")
            print("   - Upload to wrong datastore")
            print("   - Indexing errors")
        else:
            print(f"\n✅ Document count looks good: {total_docs}")
            
    except Exception as e:
        print(f"\n❌ Error accessing datastore: {e}")
        print("\nPossible issues:")
        print("1. Wrong DATASTORE_ID")
        print("2. Missing permissions")
        print("3. Datastore in different location")


def test_search_retrieval():
    """Test search retrieval to see what's being returned."""
    print("\n" + "="*60)
    print("🔎 SEARCH RETRIEVAL TEST")
    print("="*60)
    
    try:
        from google.cloud import discoveryengine_v1alpha as discoveryengine_alpha
        
        client = discoveryengine_alpha.SearchServiceClient()
        
        serving_config = f"projects/{PROJECT_ID}/locations/{LOCATION}/dataStores/{DATASTORE_ID}/servingConfigs/default_search"
        
        test_queries = [
            "neural scaling laws",
            "LLM reasoning",
            "attention mechanism transformer",
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            request = discoveryengine_alpha.SearchRequest(
                serving_config=serving_config,
                query=query,
                page_size=10,
            )
            
            response = client.search(request=request)
            
            results = list(response.results)
            print(f"   Results returned: {len(results)}")
            
            for i, result in enumerate(results[:3], 1):
                doc_id = result.document.id if result.document else "Unknown"
                print(f"   {i}. {doc_id}")
                
    except Exception as e:
        print(f"\n⚠️  Search test failed: {e}")
        print("   This might be expected if using basic tier or different API.")


def test_grounding_search():
    """Test the actual grounding search used in the app."""
    print("\n" + "="*60)
    print("🎯 GROUNDING SEARCH TEST (as used in app)")
    print("="*60)
    
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, Tool, grounding, GenerationConfig
        
        vertexai.init(project=PROJECT_ID, location=config.LOCATION)
        
        datastore_path = f"projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATASTORE_ID}"
        
        model = GenerativeModel(config.SEARCH_MODEL)
        
        vertex_search_tool = Tool.from_retrieval(
            grounding.Retrieval(
                grounding.VertexAISearch(datastore=datastore_path)
            )
        )
        
        test_query = "What are the key findings about neural scaling laws and superposition?"
        
        print(f"\n🔍 Test Query: '{test_query}'")
        print(f"📦 Datastore: {DATASTORE_ID}")
        
        response = model.generate_content(
            test_query,
            tools=[vertex_search_tool],
            generation_config=GenerationConfig(temperature=0.1, max_output_tokens=1000)
        )
        
        print(f"\n📝 Response preview: {response.text[:300]}...")
        
        # Check grounding metadata
        if response.candidates and response.candidates[0].grounding_metadata:
            metadata = response.candidates[0].grounding_metadata
            
            if metadata.grounding_chunks:
                unique_docs = set()
                for chunk in metadata.grounding_chunks:
                    if chunk.retrieved_context:
                        uri = chunk.retrieved_context.uri or ""
                        filename = uri.split("/")[-1] if uri else chunk.retrieved_context.title
                        unique_docs.add(filename)
                
                print(f"\n📚 Documents retrieved: {len(metadata.grounding_chunks)} chunks from {len(unique_docs)} unique docs")
                print(f"\n📄 Unique documents cited:")
                for i, doc in enumerate(list(unique_docs)[:15], 1):
                    print(f"   {i}. {doc}")
                
                if len(unique_docs) < 5:
                    print(f"\n⚠️  WARNING: Only {len(unique_docs)} unique documents retrieved")
                    print("   The grounding may be too narrow or datastore has limited content")
            else:
                print("\n⚠️  No grounding chunks returned!")
                print("   This suggests the datastore may not be properly connected")
        else:
            print("\n⚠️  No grounding metadata in response!")
            
    except Exception as e:
        print(f"\n❌ Grounding test failed: {e}")
        import traceback
        traceback.print_exc()


def list_all_datastores():
    """List all datastores in the project to verify which ones exist."""
    print("\n" + "="*60)
    print("📋 ALL DATASTORES IN PROJECT")
    print("="*60)
    
    try:
        client = discoveryengine.DataStoreServiceClient()
        
        # List datastores in global location
        parent = f"projects/{PROJECT_ID}/locations/global/collections/default_collection"
        
        request = discoveryengine.ListDataStoresRequest(parent=parent)
        
        print(f"\n📍 Location: global")
        for datastore in client.list_data_stores(request=request):
            ds_id = datastore.name.split("/")[-1]
            print(f"\n   📦 {ds_id}")
            print(f"      Display Name: {datastore.display_name}")
            print(f"      Industry: {datastore.industry_vertical}")
            print(f"      Solutions: {list(datastore.solution_types)}")
            
    except Exception as e:
        print(f"\n⚠️  Could not list datastores: {e}")


if __name__ == "__main__":
    print("\n" + "🔬 VERTEX AI SEARCH DATASTORE DIAGNOSTICS ".center(60, "="))
    
    # Check config first
    if not check_config():
        exit(1)
    
    print(f"\nProject: {PROJECT_ID}")
    print(f"Datastore: {DATASTORE_ID}")
    
    # Run all diagnostics
    list_all_datastores()
    check_datastore_documents()
    test_search_retrieval()
    test_grounding_search()
    
    print("\n" + "="*60)
    print("✅ DIAGNOSTICS COMPLETE")
    print("="*60)
    print("""
Next steps based on results:
1. If document count is low → Check your upload/indexing process
2. If wrong datastore → Update VERTEX_AI_DATASTORE_ID in your .env file
3. If grounding returns few docs → This is normal; Vertex AI Search 
   only returns most semantically relevant chunks per query
4. To search more docs → Run multiple diverse queries (which deep research does)
""")
