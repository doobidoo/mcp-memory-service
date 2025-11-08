#!/usr/bin/env python3
"""
Non-interactive Groq API client for AI agent integration.
Allows one AI system to call Groq's language models programmatically.
"""

import json
import os
import sys
from groq import Groq
from groq import APIError, AuthenticationError, RateLimitError, APIConnectionError


class GroqAgentBridge:
    """Bridge for other AI agents to call Groq's language models."""
    
    def __init__(self, api_key=None):
        """Initialize with API key from environment or parameter."""
        self.api_key = api_key or os.environ.get('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable required")
        
        self.client = Groq(api_key=self.api_key)
    
    def call_model(self, prompt, model="llama-3.3-70b-versatile",
                   max_tokens=1024, temperature=0.7, system_message=None):
        """
        Non-interactive call to Groq's language model.
        
        Args:
            prompt: User prompt to send to the model
            model: Model to use (default: llama-3.3-70b-versatile)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            system_message: Optional system context message
        
        Returns:
            Dict with response data or error
        """
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return {
                "status": "success",
                "response": response.choices[0].message.content,
                "model": model,
                "tokens_used": response.usage.total_tokens
            }

        except AuthenticationError as e:
            return {
                "status": "error",
                "error": f"Authentication failed: {str(e)}. Check GROQ_API_KEY environment variable.",
                "error_type": "authentication",
                "model": model
            }
        except RateLimitError as e:
            return {
                "status": "error",
                "error": f"Rate limit exceeded: {str(e)}. Please try again later.",
                "error_type": "rate_limit",
                "model": model
            }
        except APIConnectionError as e:
            return {
                "status": "error",
                "error": f"Network connection failed: {str(e)}. Check your internet connection.",
                "error_type": "connection",
                "model": model
            }
        except APIError as e:
            return {
                "status": "error",
                "error": f"Groq API error: {str(e)}",
                "error_type": "api_error",
                "model": model
            }
        except Exception as e:
            # Catch-all for unexpected errors
            return {
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
                "error_type": "unknown",
                "model": model
            }
    
    def call_model_raw(self, prompt, **kwargs):
        """Raw text response for direct consumption by other agents."""
        result = self.call_model(prompt, **kwargs)
        if result["status"] == "success":
            return result["response"]
        else:
            raise Exception(f"Groq API error: {result['error']}")


def main():
    """Command-line interface for non-interactive usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Groq API Bridge for AI Agents')
    parser.add_argument('prompt', help='Input prompt for the model')
    parser.add_argument('--model', default='llama-3.3-70b-versatile',
                       help='Model to use (default: llama-3.3-70b-versatile)')
    parser.add_argument('--max-tokens', type=int, default=1024,
                       help='Maximum tokens in response')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--system', help='System message for context')
    parser.add_argument('--json', action='store_true',
                       help='Output JSON response')
    
    args = parser.parse_args()
    
    try:
        bridge = GroqAgentBridge()
        result = bridge.call_model(
            prompt=args.prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system_message=args.system
        )
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["status"] == "success":
                print(result["response"])
            else:
                print(f"Error: {result['error']}", file=sys.stderr)
                sys.exit(1)
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()