import gradio as gr
from local_jarvis_model import LocalJarvisModel

jarvis = LocalJarvisModel()

css = """
body{
    background:#020617;
    overflow-x:hidden;
}

.gradio-container{
    background:
    radial-gradient(circle at center,
    rgba(56,189,248,.12) 0%,
    #020617 60%);
}

.reactor-wrapper{
    display:flex;
    flex-direction:column;
    align-items:center;
    margin-top:20px;
    margin-bottom:25px;
}

.reactor{
    position:relative;
    width:320px;
    height:320px;
}

.ring{
    position:absolute;
    border-radius:50%;
    border:3px solid #38bdf8;
    top:50%;
    left:50%;
    transform:translate(-50%,-50%);
}

.ring1{
    width:300px;
    height:300px;
    box-shadow:0 0 20px #38bdf8, inset 0 0 20px #38bdf8;
    animation:spin1 12s linear infinite;
}

.ring2{
    width:230px;
    height:230px;
    border-style:dashed;
    box-shadow:0 0 20px #38bdf8;
    animation:spin2 8s linear infinite reverse;
}

.ring3{
    width:160px;
    height:160px;
    box-shadow:0 0 25px #38bdf8;
    animation:spin1 6s linear infinite;
}

.core{
    position:absolute;
    width:90px;
    height:90px;
    top:50%;
    left:50%;
    transform:translate(-50%,-50%);
    border-radius:50%;
    background:radial-gradient(circle,#e0f2fe 0%,#7dd3fc 30%,#38bdf8 60%,#0284c7 100%);
    box-shadow:0 0 20px #38bdf8,0 0 50px #38bdf8,0 0 100px #38bdf8;
    animation:pulse 2s ease-in-out infinite;
}

@keyframes spin1{
    from{transform:translate(-50%,-50%) rotate(0deg);}
    to{transform:translate(-50%,-50%) rotate(360deg);}
}

@keyframes spin2{
    from{transform:translate(-50%,-50%) rotate(0deg);}
    to{transform:translate(-50%,-50%) rotate(-360deg);}
}

@keyframes pulse{
    0%{transform:translate(-50%,-50%) scale(1);}
    50%{transform:translate(-50%,-50%) scale(1.08);}
    100%{transform:translate(-50%,-50%) scale(1);}
}

.jarvis-title{
    font-size:3rem;
    font-weight:700;
    color:#7dd3fc;
    margin-top:15px;
    text-shadow:0 0 10px #38bdf8,0 0 30px #38bdf8,0 0 60px #38bdf8;
}

.jarvis-sub{
    color:#94a3b8;
    margin-top:5px;
}

button{
    border-radius:12px !important;
    border:1px solid #38bdf8 !important;
}
"""

# =========================
# CHAT FUNCTION (VERY SAFE FORMAT)
# =========================

def respond(message, history):
    if not message:
        return history, ""

    reply = jarvis.reply(message)

    history = history + [(message, reply)]
    return history, ""

# =========================
# UI
# =========================

with gr.Blocks(css=css, title="J.A.R.V.I.S") as demo:

    gr.HTML("""
    <div class="reactor-wrapper">
        <div class="reactor">
            <div class="ring ring1"></div>
            <div class="ring ring2"></div>
            <div class="ring ring3"></div>
            <div class="core"></div>
        </div>

        <div class="jarvis-title">J.A.R.V.I.S</div>
        <div class="jarvis-sub">Just A Rather Very Intelligent System</div>
    </div>
    """)

    chatbot = gr.Chatbot(height=350)

    msg = gr.Textbox(placeholder="How may I assist you, sir?")
    send = gr.Button("Send")

    send.click(
        respond,
        [msg, chatbot],
        [chatbot, msg]
    )

    msg.submit(
        respond,
        [msg, chatbot],
        [chatbot, msg]
    )

demo.launch()