# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CLI commands for document ingestion.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import click

from ..ingestion import get_loader_for_file, is_supported_file, SUPPORTED_FORMATS
from ..models.memory import Memory
from ..utils import generate_content_hash

logger = logging.getLogger(__name__)


def add_ingestion_commands(cli_group):
    """Add ingestion commands to a Click CLI group."""
    cli_group.add_command(ingest_document)
    cli_group.add_command(ingest_directory)
    cli_group.add_command(list_formats)


@click.command()
@click.argument('file_path', type=click.Path(exists=True, path_type=Path))
@click.option('--tags', '-t', multiple=True, help='Tags to apply to all memories (can be used multiple times)')
@click.option('--chunk-size', '-c', default=1000, help='Target size for text chunks in characters')
@click.option('--chunk-overlap', '-o', default=200, help='Characters to overlap between chunks')
@click.option('--memory-type', '-m', default='document', help='Type label for created memories')
@click.option('--storage-backend', '-s', default='sqlite_vec',
              type=click.Choice(['sqlite_vec', 'sqlite-vec', 'cloudflare', 'hybrid']), help='Storage backend to use')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def ingest_document(file_path: Path, tags: tuple, chunk_size: int, chunk_overlap: int,
                   memory_type: str, storage_backend: str, verbose: bool):
    """
    Ingest a single document file into the memory database.
    
    Supports multiple formats including PDF, text, and Markdown files.
    The document will be parsed, chunked intelligently, and stored as multiple memories.
    
    Examples:
        memory ingest-document manual.pdf --tags documentation,manual
        memory ingest-document README.md --chunk-size 500 --verbose
        memory ingest-document data.txt --memory-type reference --tags important
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
        click.echo(f"📄 Processing document: {file_path}")
    
    async def run_ingestion():
        from .utils import get_storage
        
        try:
            # Initialize storage
            storage = await get_storage(storage_backend)
            
            # Get appropriate document loader
            loader = get_loader_for_file(file_path)
            if loader is None:
                click.echo(f"❌ Error: Unsupported file format: {file_path.suffix}", err=True)
                return False
            
            # Configure loader
            loader.chunk_size = chunk_size
            loader.chunk_overlap = chunk_overlap
            
            if verbose:
                click.echo(f"🔧 Using loader: {loader.__class__.__name__}")
                click.echo(f"⚙️  Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
            
            start_time = time.time()
            chunks_processed = 0
            chunks_stored = 0
            errors = []
            
            # Extract and store chunks
            with click.progressbar(length=0, label='Processing chunks') as bar:
                async for chunk in loader.extract_chunks(file_path):
                    chunks_processed += 1
                    bar.length = chunks_processed
                    bar.update(1)
                    
                    try:
                        # Combine CLI tags with chunk metadata tags
                        all_tags = list(tags)
                        if chunk.metadata.get('tags'):
                            all_tags.extend(chunk.metadata['tags'])
                        
                        # Create memory object
                        memory = Memory(
                            content=chunk.content,
                            content_hash=generate_content_hash(chunk.content, chunk.metadata),
                            tags=list(set(all_tags)),  # Remove duplicates
                            memory_type=memory_type,
                            metadata=chunk.metadata
                        )
                        
                        # Store the memory
                        success, error = await storage.store(memory)
                        if success:
                            chunks_stored += 1
                        else:
                            errors.append(f"Chunk {chunk.chunk_index}: {error}")
                            if verbose:
                                click.echo(f"⚠️  Error storing chunk {chunk.chunk_index}: {error}")
                                
                    except Exception as e:
                        errors.append(f"Chunk {chunk.chunk_index}: {str(e)}")
                        if verbose:
                            click.echo(f"⚠️  Exception in chunk {chunk.chunk_index}: {str(e)}")
            
            processing_time = time.time() - start_time
            success_rate = (chunks_stored / chunks_processed * 100) if chunks_processed > 0 else 0
            
            # Display results
            click.echo(f"\n✅ Document ingestion completed: {file_path.name}")
            click.echo(f"📄 Chunks processed: {chunks_processed}")
            click.echo(f"💾 Chunks stored: {chunks_stored}")
            click.echo(f"⚡ Success rate: {success_rate:.1f}%")
            click.echo(f"⏱️  Processing time: {processing_time:.2f} seconds")
            
            if errors:
                click.echo(f"⚠️  Errors encountered: {len(errors)}")
                if verbose:
                    for error in errors[:5]:  # Show first 5 errors
                        click.echo(f"   - {error}")
                    if len(errors) > 5:
                        click.echo(f"   ... and {len(errors) - 5} more errors")
            
            return chunks_stored > 0
            
        except Exception as e:
            click.echo(f"❌ Error ingesting document: {str(e)}", err=True)
            if verbose:
                import traceback
                click.echo(traceback.format_exc(), err=True)
            return False
        finally:
            if 'storage' in locals():
                await storage.close()
    
    success = asyncio.run(run_ingestion())
    sys.exit(0 if success else 1)


@click.command()
@click.argument('directory_path', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--tags', '-t', multiple=True, help='Tags to apply to all memories (can be used multiple times)')
@click.option('--recursive', '-r', is_flag=True, default=True, help='Process subdirectories recursively')
@click.option('--extensions', '-e', multiple=True, help='File extensions to process (default: all supported)')
@click.option('--chunk-size', '-c', default=1000, help='Target size for text chunks in characters')
@click.option('--max-files', default=100, help='Maximum number of files to process')
@click.option('--storage-backend', '-s', default='sqlite_vec',
              type=click.Choice(['sqlite_vec', 'sqlite-vec', 'cloudflare', 'hybrid']), help='Storage backend to use')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--dry-run', is_flag=True, help='Show what would be processed without storing')
def ingest_directory(directory_path: Path, tags: tuple, recursive: bool, extensions: tuple,
                    chunk_size: int, max_files: int, storage_backend: str, verbose: bool, dry_run: bool):
    """
    Batch ingest all supported documents from a directory.
    
    Recursively processes all supported file types in the directory,
    creating memories with consistent tagging and metadata.
    
    Examples:
        memory ingest-directory ./docs --tags knowledge-base --recursive
        memory ingest-directory ./manuals --extensions pdf,md --max-files 50
        memory ingest-directory ./content --dry-run --verbose
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
        click.echo(f"📁 Processing directory: {directory_path}")
    
    async def run_batch_ingestion():
        from .utils import get_storage
        
        try:
            # Initialize storage (unless dry run)
            storage = None if dry_run else await get_storage(storage_backend)
            
            # Determine file extensions to process
            if extensions:
                file_extensions = [ext.lstrip('.') for ext in extensions]
            else:
                file_extensions = list(SUPPORTED_FORMATS.keys())
            
            if verbose:
                click.echo(f"🔍 Looking for extensions: {', '.join(file_extensions)}")
                click.echo(f"📊 Max files: {max_files}, Recursive: {recursive}")
            
            # Find all supported files
            all_files = []
            for ext in file_extensions:
                ext_pattern = f"*.{ext.lstrip('.')}"
                if recursive:
                    files = list(directory_path.rglob(ext_pattern))
                else:
                    files = list(directory_path.glob(ext_pattern))
                all_files.extend(files)
            
            # Remove duplicates and filter supported files
            unique_files = []
            seen = set()
            for file_path in all_files:
                if file_path not in seen and is_supported_file(file_path):
                    unique_files.append(file_path)
                    seen.add(file_path)
            
            # Limit number of files
            files_to_process = unique_files[:max_files]
            
            if not files_to_process:
                click.echo(f"❌ No supported files found in directory: {directory_path}")
                return False
            
            click.echo(f"📋 Found {len(files_to_process)} files to process")
            
            if dry_run:
                click.echo("🔍 DRY RUN - Files that would be processed:")
                for file_path in files_to_process:
                    click.echo(f"   📄 {file_path}")
                return True
            
            start_time = time.time()
            total_chunks_processed = 0
            total_chunks_stored = 0
            files_processed = 0
            files_failed = 0
            all_errors = []
            
            # Process each file with progress bar
            with click.progressbar(files_to_process, label='Processing files') as files_bar:
                for file_path in files_bar:
                    try:
                        if verbose:
                            click.echo(f"\n🔄 Processing: {file_path.name}")
                        
                        # Get appropriate document loader
                        loader = get_loader_for_file(file_path)
                        if loader is None:
                            all_errors.append(f"{file_path.name}: Unsupported format")
                            files_failed += 1
                            continue
                        
                        # Configure loader
                        loader.chunk_size = chunk_size
                        
                        file_chunks_processed = 0
                        file_chunks_stored = 0
                        
                        # Extract and store chunks from this file
                        async for chunk in loader.extract_chunks(file_path):
                            file_chunks_processed += 1
                            total_chunks_processed += 1
                            
                            try:
                                # Add directory-level tags and file-specific tags
                                all_tags = list(tags)
                                all_tags.append(f"source_dir:{directory_path.name}")
                                all_tags.append(f"file_type:{file_path.suffix.lstrip('.')}")
                                
                                if chunk.metadata.get('tags'):
                                    all_tags.extend(chunk.metadata['tags'])
                                
                                # Create memory object
                                memory = Memory(
                                    content=chunk.content,
                                    content_hash=generate_content_hash(chunk.content, chunk.metadata),
                                    tags=list(set(all_tags)),  # Remove duplicates
                                    memory_type="document",
                                    metadata=chunk.metadata
                                )
                                
                                # Store the memory
                                success, error = await storage.store(memory)
                                if success:
                                    file_chunks_stored += 1
                                    total_chunks_stored += 1
                                else:
                                    all_errors.append(f"{file_path.name} chunk {chunk.chunk_index}: {error}")
                                    
                            except Exception as e:
                                all_errors.append(f"{file_path.name} chunk {chunk.chunk_index}: {str(e)}")
                        
                        if file_chunks_stored > 0:
                            files_processed += 1
                            if verbose:
                                click.echo(f"   ✅ {file_chunks_stored}/{file_chunks_processed} chunks stored")
                        else:
                            files_failed += 1
                            if verbose:
                                click.echo(f"   ❌ No chunks stored")
                                
                    except Exception as e:
                        files_failed += 1
                        all_errors.append(f"{file_path.name}: {str(e)}")
                        if verbose:
                            click.echo(f"   ❌ Error: {str(e)}")
            
            processing_time = time.time() - start_time
            success_rate = (total_chunks_stored / total_chunks_processed * 100) if total_chunks_processed > 0 else 0
            
            # Display results
            click.echo(f"\n✅ Directory ingestion completed: {directory_path.name}")
            click.echo(f"📁 Files processed: {files_processed}/{len(files_to_process)}")
            click.echo(f"📄 Total chunks processed: {total_chunks_processed}")
            click.echo(f"💾 Total chunks stored: {total_chunks_stored}")
            click.echo(f"⚡ Success rate: {success_rate:.1f}%")
            click.echo(f"⏱️  Processing time: {processing_time:.2f} seconds")
            
            if files_failed > 0:
                click.echo(f"❌ Files failed: {files_failed}")
            
            if all_errors:
                click.echo(f"⚠️  Total errors: {len(all_errors)}")
                if verbose:
                    error_limit = 10
                    for error in all_errors[:error_limit]:
                        click.echo(f"   - {error}")
                    if len(all_errors) > error_limit:
                        click.echo(f"   ... and {len(all_errors) - error_limit} more errors")
            
            return total_chunks_stored > 0
            
        except Exception as e:
            click.echo(f"❌ Error in batch ingestion: {str(e)}", err=True)
            if verbose:
                import traceback
                click.echo(traceback.format_exc(), err=True)
            return False
        finally:
            if storage:
                await storage.close()
    
    success = asyncio.run(run_batch_ingestion())
    sys.exit(0 if success else 1)


@click.command()
def list_formats() -> None:
    """
    List all supported document formats for ingestion.
    
    Shows file extensions and descriptions of supported document types.
    """
    click.echo("📋 Supported document formats for ingestion:\n")
    
    for ext, description in SUPPORTED_FORMATS.items():
        click.echo(f"  📄 .{ext:<8} - {description}")
    
    click.echo(f"\n✨ Total: {len(SUPPORTED_FORMATS)} supported formats")
    click.echo("\nExamples:")
    click.echo("  memory ingest-document manual.pdf")
    click.echo("  memory ingest-directory ./docs --extensions pdf,md")