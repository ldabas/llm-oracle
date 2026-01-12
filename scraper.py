"""
NeurIPS Paper Scraper
Downloads papers from OpenReview (the actual source) for Vertex AI Search ingestion.
"""

import time
import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Represents a NeurIPS paper."""
    id: str
    title: str
    authors: List[str]
    abstract: str
    pdf_url: str
    openreview_url: str
    venue: str = ""  # Actual venue from OpenReview
    keywords: List[str] = None
    
    def is_neurips(self) -> bool:
        """Check if paper is actually from NeurIPS."""
        return "neurips" in self.venue.lower() or "nips" in self.venue.lower()


class OpenReviewScraper:
    """
    Scraper for OpenReview - the actual source of NeurIPS papers.
    Uses their public API which doesn't require authentication.
    """
    
    API_BASE = "https://api2.openreview.net"
    
    def __init__(self, output_dir: str = "./neurips_papers"):
        self.output_dir = Path(output_dir)
        self.pdf_dir = self.output_dir / "pdfs"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0',
        })
    
    def get_neurips_2025_papers(self, limit: int = None, accepted_only: bool = True) -> List[Dict]:
        """
        Fetch NeurIPS 2025 papers from OpenReview API.
        
        Args:
            limit: Max papers to fetch
            accepted_only: If True, filter to only accepted papers (poster/oral/spotlight)
        """
        logger.info("Fetching NeurIPS 2025 papers from OpenReview API...")
        
        papers = []
        offset = 0
        batch_size = 100
        
        while True:
            # Use the invitation-based endpoint which returns full data
            params = {
                "invitation": "NeurIPS.cc/2025/Conference/-/Submission",
                "limit": batch_size,
                "offset": offset,
            }
            
            try:
                resp = self.session.get(
                    f"{self.API_BASE}/notes",
                    params=params,
                    timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                
                batch = data.get('notes', [])
                if not batch:
                    break
                
                papers.extend(batch)
                logger.info(f"Fetched {len(papers)} papers so far...")
                
                if limit and len(papers) >= limit:
                    papers = papers[:limit]
                    break
                
                offset += batch_size
                time.sleep(0.5)  # Rate limit
                
            except requests.RequestException as e:
                logger.error(f"API request failed: {e}")
                break
        
        # Filter to accepted papers only (exclude "Submitted to" and "Rejected")
        if accepted_only:
            accepted = []
            for p in papers:
                venue = p.get('content', {}).get('venue', {})
                if isinstance(venue, dict):
                    venue = venue.get('value', '')
                # Accepted papers have "poster", "oral", or "spotlight" in venue
                if venue and ('poster' in venue.lower() or 'oral' in venue.lower() or 'spotlight' in venue.lower()):
                    accepted.append(p)
            logger.info(f"Filtered to {len(accepted)} accepted papers (from {len(papers)} total)")
            papers = accepted
        
        logger.info(f"Total papers: {len(papers)}")
        return papers
    
    def search_papers(self, query: str, limit: int = 50) -> List[Dict]:
        """
        Search for NeurIPS 2025 papers matching a query.
        
        Note: OpenReview search API is limited, so we fetch all papers
        and filter locally for better results.
        """
        logger.info(f"Searching NeurIPS 2025 papers for: {query}")
        
        # Fetch papers and filter locally (more reliable than API search)
        all_papers = self.get_neurips_2025_papers(limit=2000, accepted_only=True)
        
        query_lower = query.lower()
        matching = []
        
        for p in all_papers:
            content = p.get('content', {})
            
            # Get title and abstract
            title = content.get('title', {})
            if isinstance(title, dict):
                title = title.get('value', '')
            
            abstract = content.get('abstract', {})
            if isinstance(abstract, dict):
                abstract = abstract.get('value', '')
            
            keywords = content.get('keywords', {})
            if isinstance(keywords, dict):
                keywords = keywords.get('value', [])
            keywords_str = ' '.join(keywords) if isinstance(keywords, list) else str(keywords)
            
            # Search in title, abstract, keywords
            searchable = f"{title} {abstract} {keywords_str}".lower()
            if query_lower in searchable:
                matching.append(p)
                if len(matching) >= limit:
                    break
        
        logger.info(f"Found {len(matching)} papers matching '{query}'")
        return matching
    
    def parse_paper(self, note: Dict) -> Optional[Paper]:
        """Parse an OpenReview note into a Paper object."""
        try:
            content = note.get('content', {})
            paper_id = note.get('id', note.get('forum', 'unknown'))
            
            # Helper to extract value from OpenReview's nested format
            def get_value(field):
                val = content.get(field, '')
                if isinstance(val, dict):
                    return val.get('value', '')
                return val
            
            title = get_value('title')
            abstract = get_value('abstract')
            
            authors = content.get('authors', [])
            if isinstance(authors, dict):
                authors = authors.get('value', [])
            
            keywords = content.get('keywords', [])
            if isinstance(keywords, dict):
                keywords = keywords.get('value', [])
            
            # Get the ACTUAL venue from the response
            venue = get_value('venue')
            if not venue:
                venue = get_value('venueid')
            if not venue:
                # Check invitation field
                invitation = note.get('invitation', '')
                if 'NeurIPS' in invitation:
                    venue = "NeurIPS 2025"
            
            # Construct URLs
            forum_id = note.get('forum', paper_id)
            openreview_url = f"https://openreview.net/forum?id={forum_id}"
            pdf_url = f"https://openreview.net/pdf?id={forum_id}"
            
            paper = Paper(
                id=paper_id,
                title=title,
                authors=authors if isinstance(authors, list) else [authors],
                abstract=abstract,
                pdf_url=pdf_url,
                openreview_url=openreview_url,
                venue=venue,
                keywords=keywords if isinstance(keywords, list) else []
            )
            
            return paper
            
        except Exception as e:
            logger.warning(f"Failed to parse paper: {e}")
            return None
    
    def download_pdf(self, paper: Paper) -> Optional[Path]:
        """Download PDF for a paper."""
        # Sanitize filename
        safe_title = re.sub(r'[^\w\s-]', '', paper.title)[:60].strip()
        safe_title = safe_title.replace(' ', '_')
        filename = f"{paper.id[:20]}_{safe_title}.pdf"
        filepath = self.pdf_dir / filename
        
        if filepath.exists():
            logger.info(f"Already exists: {filename}")
            return filepath
        
        logger.info(f"Downloading: {paper.title[:50]}...")
        
        try:
            resp = self.session.get(paper.pdf_url, timeout=60)
            
            if resp.status_code == 200 and 'application/pdf' in resp.headers.get('content-type', ''):
                filepath.write_bytes(resp.content)
                logger.info(f"Saved: {filename}")
                return filepath
            else:
                logger.warning(f"PDF not available for {paper.id}")
                return None
                
        except requests.RequestException as e:
            logger.warning(f"Download failed for {paper.id}: {e}")
            return None
    
    def scrape_all(self, limit: int = None, download_pdfs: bool = True) -> List[Paper]:
        """Scrape all NeurIPS 2025 papers."""
        raw_papers = self.get_neurips_2025_papers(limit=limit)
        
        papers = []
        for raw in raw_papers:
            paper = self.parse_paper(raw)
            if paper and paper.title:
                papers.append(paper)
                
                if download_pdfs:
                    self.download_pdf(paper)
        
        # Save metadata
        self._save_metadata(papers)
        
        logger.info(f"Processed {len(papers)} papers")
        return papers
    
    def _save_metadata(self, papers: List[Paper]):
        """Save paper metadata as JSON."""
        metadata_file = self.output_dir / "papers_metadata.json"
        
        data = []
        for p in papers:
            data.append({
                "id": p.id,
                "title": p.title,
                "authors": p.authors,
                "abstract": p.abstract,
                "venue": p.venue,  # Include venue!
                "pdf_url": p.pdf_url,
                "openreview_url": p.openreview_url,
                "keywords": p.keywords or []
            })
        
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape NeurIPS papers from OpenReview")
    parser.add_argument('--output', '-o', default='./neurips_papers', help='Output directory')
    parser.add_argument('--limit', '-n', type=int, help='Max papers to fetch')
    parser.add_argument('--no-pdf', action='store_true', help='Skip PDF downloads')
    parser.add_argument('--search', '-s', help='Search query instead of fetching all')
    
    args = parser.parse_args()
    
    scraper = OpenReviewScraper(args.output)
    
    if args.search:
        # Search mode
        raw_papers = scraper.search_papers(args.search, limit=args.limit or 50)
        papers = [scraper.parse_paper(p) for p in raw_papers]
        papers = [p for p in papers if p]
        
        # Filter to only NeurIPS papers
        neurips_papers = [p for p in papers if p.is_neurips()]
        non_neurips = len(papers) - len(neurips_papers)
        
        print(f"\nFound {len(neurips_papers)} NeurIPS papers matching '{args.search}'")
        if non_neurips > 0:
            print(f"(Filtered out {non_neurips} non-NeurIPS papers)\n")
        
        for p in neurips_papers[:10]:
            print(f"• {p.title}")
            print(f"  Venue: {p.venue}")
            print(f"  PDF: {p.pdf_url}\n")
        
        if not args.no_pdf:
            for p in neurips_papers:
                scraper.download_pdf(p)
        
        papers = neurips_papers  # Use filtered list
    else:
        # Fetch all mode
        papers = scraper.scrape_all(limit=args.limit, download_pdfs=not args.no_pdf)
        
        print(f"\n✅ Scraped {len(papers)} papers")
        print(f"   PDFs: {scraper.pdf_dir}")
        print(f"   Metadata: {scraper.output_dir / 'papers_metadata.json'}")


if __name__ == "__main__":
    main()
