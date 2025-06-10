#!/usr/bin/env python3
"""
Call Center Analysis API using Amazon Bedrock Agents
"""

import os
import time
import json
import boto3
import uuid
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
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            
            self.bedrock_agent = boto3.client(
                service_name='bedrock-agent-runtime',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            
            self.agent_id = os.getenv('BEDROCK_AGENT_ID')
            self.agent_alias_id = os.getenv('BEDROCK_AGENT_ALIAS_ID')
            
            self.initialized = bool(self.agent_id and self.agent_alias_id)
        except Exception as e:
            print(f"Error initializing Bedrock: {str(e)}")
            self.initialized = False
    
    def analyze_call(self, audio_file_path: str) -> dict:
        """Analyze call audio using Bedrock agent"""
        if not self.initialized:
            return self._generate_mock_response(audio_file_path)
        
        try:
            # Create session
            session_id = str(uuid.uuid4())
            
            # Invoke agent with file path
            response = self.bedrock_agent.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=session_id,
                inputText=f"Analyze the call recording at {audio_file_path}. Provide a full analysis including summary, sentiment, intent, topics, agent performance score, and recommendations.",
                enableTrace=False
            )
            
            # Parse response
            response_text = response['completion']
            
            # Extract structured information from response text
            # In a real implementation, you would structure the agent's response
            # or use action groups for structured data
            analysis = self._parse_agent_response(response_text)
            return analysis
            
        except Exception as e:
            print(f"Error invoking Bedrock agent: {str(e)}")
            return self._generate_mock_response(audio_file_path)
    
    def _parse_agent_response(self, response_text: str) -> dict:
        """Parse the agent's response into structured data"""
        # This is a simplified parser - in production you would use
        # action groups or a more sophisticated parser
        
        # Example implementation
        try:
            # Look for JSON in response if agent returns structured data
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group(1))
                return result
            else:
                # Basic parsing logic
                lines = response_text.split('\n')
                
                analysis = {
                    "summary": "",
                    "sentiment": "neutral",
                    "sentiment_score": 0.5,
                    "intent": "general inquiry",
                    "topics": [],
                    "agent_performance_score": 5,
                    "recommendations": []
                }
                
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if "summary:" in line.lower():
                        current_section = "summary"
                        analysis["summary"] = line.split(":", 1)[1].strip()
                    elif "sentiment:" in line.lower():
                        current_section = "sentiment"
                        parts = line.split(":", 1)[1].strip().split()
                        analysis["sentiment"] = parts[0].lower()
                        # Look for score in format (0.X)
                        for part in parts:
                            if part.startswith("(") and part.endswith(")"):
                                try:
                                    analysis["sentiment_score"] = float(part[1:-1])
                                except:
                                    pass
                    elif "intent:" in line.lower():
                        current_section = "intent"
                        analysis["intent"] = line.split(":", 1)[1].strip()
                    elif "topics:" in line.lower():
                        current_section = "topics"
                    elif "performance score:" in line.lower():
                        try:
                            score_text = line.split(":", 1)[1].strip()
                            # Extract numeric value (1-10)
                            import re
                            score_match = re.search(r'(\d+)', score_text)
                            if score_match:
                                analysis["agent_performance_score"] = int(score_match.group(1))
                        except:
                            pass
                    elif "recommendations:" in line.lower():
                        current_section = "recommendations"
                    elif current_section == "topics" and line.startswith("- "):
                        analysis["topics"].append(line[2:])
                    elif current_section == "recommendations" and line.startswith("- "):
                        analysis["recommendations"].append(line[2:])
                
                return analysis
                
        except Exception as e:
            print(f"Error parsing agent response: {str(e)}")
            return self._generate_mock_response("")
    
    def _generate_mock_response(self, audio_file_path: str) -> dict:
        """Generate mock analysis for testing or when agent is unavailable"""
        return {
            "summary": f"Mock analysis for {os.path.basename(audio_file_path)}. Customer called about a billing issue and expressed frustration with recent charges.",
            "sentiment": "negative",
            "sentiment_score": 0.2,
            "intent": "billing complaint",
            "topics": ["billing", "account charges", "refund request"],
            "agent_performance_score": 6,
            "recommendations": [
                "Use more empathetic language when addressing billing concerns",
                "Explain fee structure more clearly",
                "Offer proactive solutions before customer asks"
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
    uvicorn.run("call_center_analysis_api:app", host="0.0.0.0", port=8000, reload=False)