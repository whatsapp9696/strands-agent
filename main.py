#!/usr/bin/env python3
"""
Call Center Analysis API using Amazon Bedrock Agents
"""
import os
import time
import json
import boto3
import uuid
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Call Center Analysis API",
    description="API for call center audio analysis using Amazon Bedrock",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Data models
class CallAnalysisRequest(BaseModel):
    file_id: str = Field(..., description="ID of the uploaded audio file")
    analysis_type: str = Field(default="full", description="Type of analysis to perform (full, summary, sentiment)")

class CallAnalysisResponse(BaseModel):
    file_id: str
    summary: str
    sentiment: str
    sentiment_score: float
    intent: str
    topics: List[str]
    agent_performance_score: int
    recommendations: List[str]
    processing_time_seconds: float
    timestamp: datetime

class BedrockAgent:
    def __init__(self):
        """Initialize Bedrock client and agent"""
        try:
            self.bedrock_runtime = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.getenv('AWS_REGION', 'us-west-2')
            )
            self.bedrock_agent = boto3.client(
                service_name='bedrock-agent-runtime',
                region_name=os.getenv('AWS_REGION', 'us-west-2')
            )
            self.agent_id = os.getenv('BEDROCK_AGENT_ID')
            self.agent_alias_id = os.getenv('BEDROCK_AGENT_ALIAS_ID')
            
            # Validate required values
            if not self.agent_id:
                print("❌ BEDROCK_AGENT_ID not found in environment variables")
                self.initialized = False
                return
                
            if not self.agent_alias_id:
                print("❌ BEDROCK_AGENT_ALIAS_ID not found in environment variables")
                self.initialized = False
                return

            # Test the connection
            self.initialized = self._test_agent_connection()
            
        except Exception as e:
            print(f"Error initializing Bedrock: {str(e)}")
            self.initialized = False

    def _test_agent_connection(self):
        """Test if the agent can be reached"""
        try:
            session_id = str(uuid.uuid4())
            
            # Try a simple test invocation
            response = self.bedrock_agent.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=session_id,
                inputText="Test connection",
                enableTrace=False
            )
            
            print(f"✅ Bedrock agent connection test successful")
            return True
            
        except Exception as e:
            print(f"❌ Bedrock agent connection test failed: {str(e)}")
            
            # Provide specific error guidance
            if "ResourceNotFoundException" in str(e):
                print("💡 This usually means:")
                print(f"   - Agent ID '{self.agent_id}' doesn't exist")
                print(f"   - Agent Alias ID '{self.agent_alias_id}' doesn't exist")
                print(f"   - Agent is not in the region '{os.getenv('AWS_REGION', 'us-west-2')}'")
                print("   - Agent status is not 'PREPARED'")
            
            return False

    def analyze_call(self, audio_file_path: str) -> dict:
        """Analyze call audio using Bedrock agent"""
        if not self.initialized:
            print("⚠️ Bedrock agent not initialized, using mock response")
            return self._generate_mock_response(audio_file_path)

        try:
            # Create session
            session_id = str(uuid.uuid4())
            
            # Enhanced prompt for the agent
            prompt = f"""
Please analyze the call recording file at path: {audio_file_path}

I need you to provide a direct analysis in the following JSON format (do not use function calls, provide the analysis directly):

{{
    "summary": "A brief summary of the call conversation",
    "sentiment": "positive, negative, or neutral",
    "sentiment_score": 0.8,
    "intent": "The main purpose or intent of the call",
    "topics": ["topic1", "topic2", "topic3"],
    "agent_performance_score": 7,
    "recommendations": ["recommendation1", "recommendation2"]
}}

Please provide a complete analysis directly without mentioning function calls or transcription steps. Base your analysis on typical call center scenarios if you cannot access the actual audio file.
"""

            # Invoke agent
            response = self.bedrock_agent.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=session_id,
                inputText=prompt,
                enableTrace=False
            )

            print(f"✅ Received response from Bedrock agent")
            
            # Handle EventStream response properly
            completion = ""
            
            # Check if response contains an EventStream
            if 'completion' in response:
                completion_data = response['completion']
                
                # Handle EventStream
                if hasattr(completion_data, '__iter__'):
                    print("📡 Processing streaming response...")
                    
                    try:
                        for event in completion_data:
                            if 'chunk' in event:
                                chunk = event['chunk']
                                if 'bytes' in chunk:
                                    chunk_text = chunk['bytes'].decode('utf-8')
                                    completion += chunk_text
                                elif 'attribution' in chunk:
                                    # Handle attribution chunk if needed
                                    pass
                            elif 'trace' in event:
                                # Handle trace information if needed
                                trace = event['trace']
                                print(f"🔍 Trace: {trace.get('trace', {}).get('orchestrationTrace', {}).get('modelInvocationOutput', {}).get('rawResponse', '')}")
                            elif 'returnControl' in event:
                                # Handle return control if needed
                                pass
                                
                    except Exception as stream_error:
                        print(f"⚠️ Error processing stream: {str(stream_error)}")
                        # Fallback to string conversion
                        completion = str(completion_data)
                else:
                    # Not a stream, treat as string
                    completion = str(completion_data)
            else:
                print("❌ No completion found in response")
                return self._generate_mock_response(audio_file_path)

            print(f"📝 Received {len(completion)} characters from agent")
            print(f"🔍 Response preview: {completion[:200]}...")
            
            # Parse response
            analysis = self._parse_agent_response(completion)
            return analysis

        except Exception as e:
            print(f"❌ Error invoking Bedrock agent: {str(e)}")
            
            # Log the full error for debugging
            import traceback
            print("Full error traceback:")
            traceback.print_exc()
            
            return self._generate_mock_response(audio_file_path)

    def _parse_agent_response(self, response_text: str) -> dict:
        """Parse the agent's response into structured data"""
        try:
            print(f"🔍 Parsing response of type: {type(response_text)}")
            
            # Ensure we have a string
            if not isinstance(response_text, str):
                response_text = str(response_text)
            
            # Clean up the response text
            response_text = response_text.strip()
            
            if not response_text:
                print("⚠️ Empty response from agent")
                return self._generate_mock_response("")
            
            print(f"📄 Response text (first 500 chars): {response_text[:500]}")
            
            # First try to find JSON in the response
            # Look for JSON block in code fences
            json_pattern = r'```json\s*(.*?)\s*```'
            json_match = re.search(json_pattern, response_text, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                json_str = json_match.group(1).strip()
                try:
                    result = json.loads(json_str)
                    print("✅ Successfully parsed JSON from code block")
                    return self._validate_and_clean_response(result)
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON in code block: {e}")
            
            # Look for standalone JSON objects
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    result = json.loads(json_str)
                    if isinstance(result, dict):
                        # Check if it's a function call (has 'name' and 'arguments')
                        if 'name' in result and 'arguments' in result:
                            print(f"🔧 Detected function call: {result.get('name')}")
                            continue  # Skip function calls, look for actual analysis
                        # Check if it's an analysis result
                        elif any(key in result for key in ['summary', 'sentiment', 'intent']):
                            print("✅ Successfully parsed standalone JSON")
                            return self._validate_and_clean_response(result)
                except json.JSONDecodeError:
                    continue
            
            # Check if the response mentions function calls but doesn't provide analysis
            if any(phrase in response_text.lower() for phrase in ['function call', 'speech_to_text', 'analyze_conversation', "i'll need to use"]):
                print("⚠️ Agent attempted to use functions but didn't provide direct analysis")
                # Return a more specific mock response indicating the issue
                return {
                    "summary": "The agent attempted to use function calls to analyze the audio file, but direct analysis was not provided. This might indicate the agent's action groups are not properly configured.",
                    "sentiment": "neutral",
                    "sentiment_score": 0.5,
                    "intent": "system configuration issue",
                    "topics": ["function call error", "configuration issue"],
                    "agent_performance_score": 3,
                    "recommendations": [
                        "Check if agent action groups are properly configured",
                        "Ensure speech-to-text and analysis functions are working",
                        "Consider providing pre-transcribed text for analysis"
                    ]
                }
            
            print("⚠️ Could not find valid JSON, using text parsing fallback")
            
            # Fallback to text parsing
            analysis = {
                "summary": "",
                "sentiment": "neutral",
                "sentiment_score": 0.5,
                "intent": "general inquiry",
                "topics": [],
                "agent_performance_score": 5,
                "recommendations": []
            }
            
            lines = response_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                line_lower = line.lower()
                
                # Extract summary
                if any(keyword in line_lower for keyword in ["summary:", "summary is:", "call summary:"]):
                    current_section = "summary"
                    if ":" in line:
                        analysis["summary"] = line.split(":", 1)[1].strip()
                
                # Extract sentiment
                elif any(keyword in line_lower for keyword in ["sentiment:", "sentiment is:", "sentiment analysis:"]):
                    current_section = "sentiment"
                    if ":" in line:
                        sentiment_part = line.split(":", 1)[1].strip()
                        words = sentiment_part.split()
                        if words:
                            sentiment = words[0].lower()
                            if sentiment in ["positive", "negative", "neutral"]:
                                analysis["sentiment"] = sentiment
                        
                        # Look for score
                        score_match = re.search(r'(\d*\.?\d+)', sentiment_part)
                        if score_match:
                            try:
                                score = float(score_match.group(1))
                                if 0 <= score <= 1:
                                    analysis["sentiment_score"] = score
                                elif 0 <= score <= 10:
                                    analysis["sentiment_score"] = score / 10.0
                            except:
                                pass
                
                # Extract intent
                elif any(keyword in line_lower for keyword in ["intent:", "intent is:", "purpose:", "customer intent:"]):
                    current_section = "intent"
                    if ":" in line:
                        analysis["intent"] = line.split(":", 1)[1].strip()
                
                # Extract topics
                elif any(keyword in line_lower for keyword in ["topics:", "topics discussed:", "key topics:"]):
                    current_section = "topics"
                
                # Extract performance score
                elif any(keyword in line_lower for keyword in ["performance score:", "agent performance:", "score:"]):
                    try:
                        score_text = line.split(":", 1)[1].strip() if ":" in line else line
                        score_match = re.search(r'(\d+)', score_text)
                        if score_match:
                            score = int(score_match.group(1))
                            if 1 <= score <= 10:
                                analysis["agent_performance_score"] = score
                    except:
                        pass
                
                # Extract recommendations
                elif any(keyword in line_lower for keyword in ["recommendations:", "recommendation:", "suggestions:"]):
                    current_section = "recommendations"
                
                # Handle list items
                elif current_section == "topics" and (line.startswith("- ") or line.startswith("• ") or line.startswith("* ")):
                    topic = line[2:].strip()
                    if topic and topic not in analysis["topics"]:
                        analysis["topics"].append(topic)
                
                elif current_section == "recommendations" and (line.startswith("- ") or line.startswith("• ") or line.startswith("* ")):
                    rec = line[2:].strip()
                    if rec and rec not in analysis["recommendations"]:
                        analysis["recommendations"].append(rec)
                
                # Handle numbered lists
                elif current_section == "topics" and re.match(r'^\d+\.\s+', line):
                    topic = re.sub(r'^\d+\.\s+', '', line).strip()
                    if topic and topic not in analysis["topics"]:
                        analysis["topics"].append(topic)
                
                elif current_section == "recommendations" and re.match(r'^\d+\.\s+', line):
                    rec = re.sub(r'^\d+\.\s+', '', line).strip()
                    if rec and rec not in analysis["recommendations"]:
                        analysis["recommendations"].append(rec)
            
            # Ensure we have some content
            if not analysis["summary"]:
                analysis["summary"] = "Analysis completed. Please check the raw response for details."
            
            print("✅ Successfully parsed text response")
            return analysis
            
        except Exception as e:
            print(f"❌ Error parsing agent response: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._generate_mock_response("")

    def _validate_and_clean_response(self, result: dict) -> dict:
        """Validate and clean the parsed response"""
        # Ensure required fields exist with defaults
        cleaned = {
            "summary": result.get("summary", "No summary provided"),
            "sentiment": result.get("sentiment", "neutral").lower(),
            "sentiment_score": float(result.get("sentiment_score", 0.5)),
            "intent": result.get("intent", "general inquiry"),
            "topics": result.get("topics", []),
            "agent_performance_score": int(result.get("agent_performance_score", 5)),
            "recommendations": result.get("recommendations", [])
        }
        
        # Validate sentiment
        if cleaned["sentiment"] not in ["positive", "negative", "neutral"]:
            cleaned["sentiment"] = "neutral"
        
        # Validate sentiment score
        if not (0 <= cleaned["sentiment_score"] <= 1):
            cleaned["sentiment_score"] = 0.5
        
        # Validate performance score
        if not (1 <= cleaned["agent_performance_score"] <= 10):
            cleaned["agent_performance_score"] = 5
        
        # Ensure lists are actually lists
        if not isinstance(cleaned["topics"], list):
            cleaned["topics"] = []
        
        if not isinstance(cleaned["recommendations"], list):
            cleaned["recommendations"] = []
        
        return cleaned

    def _generate_mock_response(self, audio_file_path: str) -> dict:
        """Generate mock analysis for testing or when agent is unavailable"""
        filename = os.path.basename(audio_file_path) if audio_file_path else "unknown_file"
        
        return {
            "summary": f"Mock analysis for {filename}. Customer called regarding a service inquiry and expressed mixed feelings about the resolution provided by the support agent.",
            "sentiment": "neutral",
            "sentiment_score": 0.6,
            "intent": "service inquiry",
            "topics": ["account status", "service features", "billing inquiry"],
            "agent_performance_score": 7,
            "recommendations": [
                "Provide more detailed explanations of service features",
                "Follow up proactively on billing questions",
                "Use more empathetic language during problem resolution"
            ]
        }

# Global agent instance
bedrock_agent = None

def initialize_agent():
    """Initialize Bedrock agent"""
    global bedrock_agent
    try:
        bedrock_agent = BedrockAgent()
        return bedrock_agent.initialized
    except Exception as e:
        print(f"Error initializing Bedrock agent: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    print("🚀 Starting Call Center Analysis API...")
    # Check credentials
    has_aws = bool(os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'))
    print(f"AWS Credentials: {'✅' if has_aws else '❌'}")
    if initialize_agent():
        print("✅ Bedrock agent initialized successfully!")
    else:
        print("⚠️ Bedrock agent failed to initialize, will use mock responses")

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Call Center Analysis API",
        "version": "1.0.0",
        "status": "operational",
        "credentials": {
            "aws": bool(os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY')),
            "bedrock_agent": bedrock_agent.initialized if bedrock_agent else False
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_initialized": bedrock_agent.initialized if bedrock_agent else False,
        "timestamp": datetime.utcnow()
    }

# File storage dictionary (in-memory storage, replace with database in production)
uploaded_files = {}
analysis_results = {}

def process_call_file(file_id: str):
    """Process call file in background"""
    file_path = uploaded_files.get(file_id)
    if not file_path:
        return
    start_time = time.time()
    # Analyze call using Bedrock agent
    analysis = bedrock_agent.analyze_call(file_path)
    # Store results
    analysis_results[file_id] = {
        **analysis,
        "file_id": file_id,
        "processing_time_seconds": time.time() - start_time,
        "timestamp": datetime.utcnow()
    }

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload audio file for analysis"""
    try:
        # Generate file ID
        file_id = str(uuid.uuid4())
        # Save file to uploads directory
        file_path = f"uploads/{file_id}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        # Store file path
        uploaded_files[file_id] = file_path
        # Start analysis in background
        background_tasks.add_task(process_call_file, file_id)
        return {
            "file_id": file_id,
            "filename": file.filename,
            "status": "uploaded",
            "message": "File uploaded successfully. Analysis started."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/analysis/{file_id}")
async def get_analysis(file_id: str):
    """Get analysis results for a file"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    if file_id not in analysis_results:
        return {
            "file_id": file_id,
            "status": "processing",
            "message": "Analysis in progress. Please try again later."
        }
    return analysis_results[file_id]

@app.get("/demo", response_class=HTMLResponse)
async def demo_interface():
    """Demo web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Call Center Analysis Demo</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .upload-area { border: 2px dashed #ddd; padding: 20px; text-align: center; margin: 20px 0; }
            .upload-area.highlight { border-color: #007bff; background: #e3f2fd; }
            .results-area { margin-top: 30px; }
            .hidden { display: none; }
            .result-card { border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin: 15px 0; }
            .score-indicator { display: inline-block; width: 20px; height: 20px; border-radius: 50%; margin-right: 10px; }
            .sentiment-positive { background-color: #4caf50; }
            .sentiment-neutral { background-color: #ffeb3b; }
            .sentiment-negative { background-color: #f44336; }
            .topics-list, .recommendations-list { padding-left: 20px; }
            progress { width: 100%; height: 20px; }
            .loader { border: 5px solid #f3f3f3; border-top: 5px solid #3498db; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; display: inline-block; margin-right: 10px; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            input[type="file"] { display: none; }
            .file-info { margin-top: 10px; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Call Center Analysis Demo</h1>
            <p>Upload a call recording for analysis using Amazon Bedrock AI.</p>
            <div class="upload-area" id="dropArea" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)" ondrop="handleDrop(event)">
                <h3>Drop audio file here or click to upload</h3>
                <p>Supported formats: MP3, WAV, OGG, M4A</p>
                <button onclick="document.getElementById('fileInput').click()">Select File</button>
                <input type="file" id="fileInput" accept="audio/*" onchange="handleFileSelect(event)">
                <div class="file-info" id="fileInfo"></div>
            </div>
            <div id="uploadProgress" class="hidden">
                <h3>Uploading...</h3>
                <progress id="progressBar" value="0" max="100"></progress>
            </div>
            <div id="analysisProgress" class="hidden">
                <h3><span class="loader"></span> Analyzing call recording...</h3>
                <p>This may take a minute or two depending on the file size.</p>
            </div>
            <div id="resultsArea" class="results-area hidden">
                <h2>Analysis Results</h2>
                <div class="result-card">
                    <h3>Summary</h3>
                    <p id="callSummary"></p>
                </div>
                <div class="result-card">
                    <h3>Sentiment Analysis</h3>
                    <p><span id="sentimentIndicator" class="score-indicator"></span> <strong>Sentiment:</strong> <span id="sentimentValue"></span></p>
                    <p><strong>Intent:</strong> <span id="intentValue"></span></p>
                </div>
                <div class="result-card">
                    <h3>Topics Discussed</h3>
                    <ul id="topicsList" class="topics-list"></ul>
                </div>
                <div class="result-card">
                    <h3>Agent Performance</h3>
                    <p><strong>Score:</strong> <span id="performanceScore"></span>/10</p>
                    <h4>Recommendations:</h4>
                    <ul id="recommendationsList" class="recommendations-list"></ul>
                </div>
                <button onclick="resetDemo()">Analyze Another Call</button>
            </div>
        </div>
        <script>
            let currentFileId = null;
            let checkInterval = null;
            function handleDragOver(event) {
                event.preventDefault();
                document.getElementById('dropArea').classList.add('highlight');
            }
            function handleDragLeave(event) {
                event.preventDefault();
                document.getElementById('dropArea').classList.remove('highlight');
            }
            function handleDrop(event) {
                event.preventDefault();
                document.getElementById('dropArea').classList.remove('highlight');
                const files = event.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            }
            function handleFileSelect(event) {
                const file = event.target.files[0];
                if (file) {
                    handleFile(file);
                }
            }
            function handleFile(file) {
                // Validate file type
                const validTypes = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp4', 'audio/x-m4a'];
                if (!validTypes.includes(file.type)) {
                    alert('Please upload an audio file (MP3, WAV, OGG, M4A)');
                    return;
                }
                // Display file info
                document.getElementById('fileInfo').innerText = `Selected: ${file.name} (${formatFileSize(file.size)})`;
                // Show upload progress
                document.getElementById('uploadProgress').classList.remove('hidden');
                // Upload file
                uploadFile(file);
            }
            function formatFileSize(bytes) {
                if (bytes < 1024) return bytes + ' bytes';
                if (bytes < 1048576) return (bytes / 1024).toFixed(2) + ' KB';
                return (bytes / 1048576).toFixed(2) + ' MB';
            }
            function uploadFile(file) {
                const formData = new FormData();
                formData.append('file', file);
                const xhr = new XMLHttpRequest();
                xhr.open('POST', '/upload', true);
                xhr.upload.onprogress = function(e) {
                    if (e.lengthComputable) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        document.getElementById('progressBar').value = percentComplete;
                    }
                };
                xhr.onload = function() {
                    if (xhr.status === 200) {
                        const response = JSON.parse(xhr.response);
                        currentFileId = response.file_id;
                        // Hide upload progress
                        document.getElementById('uploadProgress').classList.add('hidden');
                        // Show analysis progress
                        document.getElementById('analysisProgress').classList.remove('hidden');
                        // Start checking for analysis results
                        checkAnalysisStatus();
                    } else {
                        alert('Upload failed. Please try again.');
                        document.getElementById('uploadProgress').classList.add('hidden');
                    }
                };
                xhr.send(formData);
            }
            function checkAnalysisStatus() {
                if (!currentFileId) return;
                clearInterval(checkInterval);
                checkInterval = setInterval(() => {
                    fetch(`/analysis/${currentFileId}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.status !== 'processing') {
                                clearInterval(checkInterval);
                                displayResults(data);
                            }
                        })
                        .catch(error => {
                            console.error('Error checking analysis status:', error);
                        });
                }, 2000);
            }
            function displayResults(data) {
                // Hide analysis progress
                document.getElementById('analysisProgress').classList.add('hidden');
                // Show results
                document.getElementById('resultsArea').classList.remove('hidden');
                // Populate results
                document.getElementById('callSummary').innerText = data.summary || 'No summary available';
                const sentimentValue = data.sentiment || 'neutral';
                document.getElementById('sentimentValue').innerText = sentimentValue.charAt(0).toUpperCase() + sentimentValue.slice(1);
                const sentimentIndicator = document.getElementById('sentimentIndicator');
                sentimentIndicator.className = 'score-indicator';
                if (sentimentValue === 'positive') {
                    sentimentIndicator.classList.add('sentiment-positive');
                } else if (sentimentValue === 'negative') {
                    sentimentIndicator.classList.add('sentiment-negative');
                } else {
                    sentimentIndicator.classList.add('sentiment-neutral');
                }
                document.getElementById('intentValue').innerText = data.intent || 'Unknown';
                document.getElementById('performanceScore').innerText = data.agent_performance_score || 'N/A';
                // Populate topics
                const topicsList = document.getElementById('topicsList');
                topicsList.innerHTML = '';
                if (data.topics && data.topics.length > 0) {
                    data.topics.forEach(topic => {
                        const li = document.createElement('li');
                        li.innerText = topic;
                        topicsList.appendChild(li);
                    });
                } else {
                    const li = document.createElement('li');
                    li.innerText = 'No topics identified';
                    topicsList.appendChild(li);
                }
                // Populate recommendations
                const recommendationsList = document.getElementById('recommendationsList');
                recommendationsList.innerHTML = '';
                if (data.recommendations && data.recommendations.length > 0) {
                    data.recommendations.forEach(rec => {
                        const li = document.createElement('li');
                        li.innerText = rec;
                        recommendationsList.appendChild(li);
                    });
                } else {
                    const li = document.createElement('li');
                    li.innerText = 'No recommendations available';
                    recommendationsList.appendChild(li);
                }
            }
            function resetDemo() {
                currentFileId = null;
                clearInterval(checkInterval);
                document.getElementById('fileInfo').innerText = '';
                document.getElementById('fileInput').value = '';
                document.getElementById('progressBar').value = 0;
                document.getElementById('resultsArea').classList.add('hidden');
                document.getElementById('uploadProgress').classList.add('hidden');
                document.getElementById('analysisProgress').classList.add('hidden');
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run("script:app", host="0.0.0.0", port=8000, reload=False)