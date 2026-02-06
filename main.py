import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

# 1. Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory databases
calls_database = []
appointments_database = []

# --- AI ANALYSIS FUNCTION ---
def analyze_with_ai(transcript):
    """Sends transcript to Llama-3 via Groq for analysis"""
    if not GROQ_API_KEY:
        print("⚠️ NO API KEY FOUND. Using fallback logic.")
        return {"summary": "AI Key Missing", "urgency": "ROUTINE", "category": "General"}

    system_prompt = """
    You are a medical triage AI. Analyze this patient call transcript.
    Return ONLY a JSON object with these 3 fields:
    1. "summary": A 1-sentence medical summary of the patient's issue.
    2. "urgency": One of ["EMERGENCY", "PRIORITY", "ROUTINE"].
       - EMERGENCY: Life-threatening (chest pain, breathing issues, stroke symptoms).
       - PRIORITY: Urgent but stable (fever, fracture, severe pain).
       - ROUTINE: Checkups, mild symptoms, inquiries.
    3. "category": A 1-2 word tag (e.g., "Cardiology", "General", "Pediatrics").
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript: {transcript}"}
            ],
            temperature=0.5,
            max_completion_tokens=200,
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        result = json.loads(completion.choices[0].message.content)
        return result
    except Exception as e:
        print(f"❌ Groq Error: {e}")
        return {"summary": "Analysis failed", "urgency": "ROUTINE", "category": "Unknown"}

# --- ROUTES ---

@app.post("/webhook/vapi")
async def receive_vapi_webhook(request: Request):
    data = await request.json()
    message_type = data.get("message", {}).get("type", "")

    if message_type != "end-of-call-report":
        return {"status": "skipped"}

    # Extract Basic Data
    call_data = data.get("message", {})
    phone = "Unknown"
    if "call" in data:
        customer = data["call"].get("customer", {})
        phone = customer.get("number") or customer.get("phoneNumber") or phone

    # Build Transcript
    raw_transcript = call_data.get("transcript", "")
    if isinstance(raw_transcript, list):
        transcript = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in raw_transcript])
    else:
        transcript = str(raw_transcript)

    # --- ⚡ AI MAGIC HAPPENS HERE ---
    print(f"🤖 Analyzing call from {phone} with Groq...")
    ai_analysis = analyze_with_ai(transcript)
    
    call_record = {
        "id": call_data.get("callId") or str(len(calls_database) + 1),
        "phone": phone,
        "transcript": transcript,
        "summary": ai_analysis["summary"],       # AI Generated
        "urgency": ai_analysis["urgency"],       # AI Generated
        "category": ai_analysis["category"],     # AI Generated
        "duration": call_data.get("duration", 0),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "called_back": False
    }

    calls_database.append(call_record)
    print(f"✅ Saved Analysis: {ai_analysis['urgency']} - {ai_analysis['category']}")
    return {"status": "success"}

@app.get("/api/calls")
def get_calls():
    return calls_database

@app.get("/api/appointments")
def get_appointments():
    return appointments_database

@app.post("/api/appointments")
async def create_appointment(request: Request):
    data = await request.json()
    appt = {
        "id": str(len(appointments_database) + 1),
        "title": f"Callback: {data['phone']}",
        "start": f"{data['date']}T{data['time']}", # ISO format for Calendar
        "patient": data['phone'],
        "notes": data.get('notes', ''),
        "type": "callback"
    }
    appointments_database.append(appt)
    
    # Update call status
    for c in calls_database:
        if c['id'] == data.get('call_id'):
            c['called_back'] = True
            
    return {"status": "success", "appointment": appt}

@app.post("/api/calls/{call_id}/callback")
def mark_callback(call_id: str):
    for c in calls_database:
        if c['id'] == call_id:
            c['called_back'] = True
    return {"status": "updated"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
