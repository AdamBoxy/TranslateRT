import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Camera, Mic, Globe, X, FlipHorizontal, Download, Scan } from 'lucide-react';
import { GoogleGenAI, Modality } from '@google/genai';

class AudioRecorder {
  context: AudioContext;
  stream: MediaStream;
  processor: ScriptProcessorNode;
  source: MediaStreamAudioSourceNode;
  onData: (base64: string) => void;

  constructor(stream: MediaStream, onData: (base64: string) => void) {
    this.stream = stream;
    this.onData = onData;
    this.context = new AudioContext({ sampleRate: 16000 });
    this.source = this.context.createMediaStreamSource(stream);
    this.processor = this.context.createScriptProcessor(4096, 1, 1);
    
    this.processor.onaudioprocess = (e) => {
      const inputData = e.inputBuffer.getChannelData(0);
      const pcm16 = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        let s = Math.max(-1, Math.min(1, inputData[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      const buffer = new ArrayBuffer(pcm16.length * 2);
      const view = new DataView(buffer);
      for (let i = 0; i < pcm16.length; i++) {
        view.setInt16(i * 2, pcm16[i], true);
      }
      
      let binary = '';
      const bytes = new Uint8Array(buffer);
      const chunkSize = 0x8000;
      for (let i = 0; i < bytes.length; i += chunkSize) {
        binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + chunkSize)));
      }
      this.onData(btoa(binary));
    };

    this.source.connect(this.processor);
    this.processor.connect(this.context.destination);
  }

  stop() {
    this.processor.disconnect();
    this.source.disconnect();
    this.context.close();
  }
}

class AudioPlayer {
  context: AudioContext;
  nextPlayTime: number;

  constructor() {
    this.context = new AudioContext({ sampleRate: 24000 });
    this.nextPlayTime = this.context.currentTime;
  }

  play(base64: string) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    const pcm16 = new Int16Array(bytes.buffer);
    const float32 = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i++) {
      float32[i] = pcm16[i] / 32768;
    }

    const buffer = this.context.createBuffer(1, float32.length, 24000);
    buffer.copyToChannel(float32, 0);

    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);

    const startTime = Math.max(this.nextPlayTime, this.context.currentTime);
    source.start(startTime);
    this.nextPlayTime = startTime + buffer.duration;
  }

  stop() {
    this.context.close();
  }
}

const HELLOS = ["Hello", "Hola", "Bonjour", "Hallo", "こんにちは", "你好", "नमस्ते", "مرحبا"];

function IntroScreen({ onComplete }: { onComplete: () => void }) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (index < HELLOS.length) {
      const timer = setTimeout(() => setIndex(index + 1), 500);
      return () => clearTimeout(timer);
    } else {
      onComplete();
    }
  }, [index, onComplete]);

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-stone-50 to-stone-100 flex items-center justify-center">
      <AnimatePresence mode="wait">
        {index < HELLOS.length ? (
          <motion.h1
            key={index}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="text-stone-800 text-5xl md:text-7xl font-light tracking-tight italic"
          >
            {HELLOS[index]}
          </motion.h1>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

function ReadyScreen({ onStart }: { onStart: () => void }) {
  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="fixed inset-0 bg-gradient-to-br from-stone-50 to-stone-100 flex flex-col items-center justify-center"
    >
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={onStart}
        className="bg-white text-stone-800 rounded-[3rem] w-44 h-44 flex flex-col items-center justify-center shadow-[0_20px_50px_rgba(0,0,0,0.05)] border border-white transition-all hover:shadow-[0_20px_50px_rgba(0,0,0,0.1)]"
      >
        <span className="text-4xl font-light mb-4 tracking-tight">Start</span>
        <div className="flex gap-3 text-stone-300">
          <Camera size={24} strokeWidth={1} />
          <Mic size={24} strokeWidth={1} />
        </div>
      </motion.button>
      <p className="text-stone-400 mt-12 text-xs font-semibold tracking-[0.2em] uppercase">Ready to translate</p>
    </motion.div>
  );
}

function LiveSession({ 
  targetLang, 
  setTargetLang,
  facingMode, 
  setFacingMode, 
  onStop 
}: { 
  targetLang: string, 
  setTargetLang: (l: string) => void,
  facingMode: 'user' | 'environment', 
  setFacingMode: (m: 'user' | 'environment' | ((prev: 'user' | 'environment') => 'user' | 'environment')) => void, 
  onStop: () => void 
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [captions, setCaptions] = useState<string>("");
  const [isConnecting, setIsConnecting] = useState(true);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [imageSummary, setImageSummary] = useState<string | null>(null);
  const [isSnapping, setIsSnapping] = useState(false);
  const sessionRef = useRef<any>(null);
  const audioRecorderRef = useRef<AudioRecorder | null>(null);
  const audioPlayerRef = useRef<AudioPlayer | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captionTimeoutRef = useRef<any>(null);

  useEffect(() => {
    let active = true;
    async function setup() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode },
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          }
        });
        if (!active) {
          stream.getTracks().forEach(t => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        const ai = new GoogleGenAI({ apiKey: (import.meta as any).env?.VITE_GEMINI_API_KEY || process.env.GEMINI_API_KEY });
        
        audioPlayerRef.current = new AudioPlayer();
        
        const sessionPromise = ai.live.connect({
          model: "gemini-2.5-flash-native-audio-preview-09-2025",
          config: {
            responseModalities: [Modality.AUDIO],
            systemInstruction: `You are a friendly real-time translation assistant. 
The user is streaming video and audio. Translate everything into ${targetLang}.
CRITICAL: Keep translations extremely short and readable. 
Use simple words. Output ONLY the translation. 
No descriptions, no "The text says...", just the translation.`,
            temperature: 0.5,
            speechConfig: {
              voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } }
            },
            outputAudioTranscription: {},
            inputAudioTranscription: {},
          },
          callbacks: {
            onopen: () => {
              if (!active) return;
              setIsConnecting(false);
              audioRecorderRef.current = new AudioRecorder(stream, (base64) => {
                sessionPromise.then(session => {
                  session.sendRealtimeInput({
                    media: {
                      mimeType: "audio/pcm;rate=16000",
                      data: base64
                    }
                  });
                });
              });
            },
            onmessage: (msg: any) => {
              if (!active) return;
              
              // Handle audio
              const parts = msg.serverContent?.modelTurn?.parts;
              if (parts) {
                for (const part of parts) {
                  if (part.inlineData && part.inlineData.data) {
                    audioPlayerRef.current?.play(part.inlineData.data);
                  }
                  if (part.text) {
                    // For short texts, we just show the latest translation
                    setCaptions(part.text);
                    if (captionTimeoutRef.current) clearTimeout(captionTimeoutRef.current);
                    captionTimeoutRef.current = setTimeout(() => setCaptions(""), 4000);
                  }
                }
              }
            },
            onclose: () => {
              if (active) onStop();
            },
            onerror: (err) => {
              console.error("Live API Error:", err);
            }
          }
        });

        sessionRef.current = sessionPromise;

        const sendVideoFrame = () => {
          if (!active || !videoRef.current || !canvasRef.current) return;
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            canvas.width = 640;
            canvas.height = (640 / video.videoWidth) * video.videoHeight;
            const ctx = canvas.getContext('2d');
            if (ctx) {
              ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
              const base64 = canvas.toDataURL('image/jpeg', 0.5).split(',')[1];
              sessionPromise.then(session => {
                session.sendRealtimeInput({
                  media: {
                    mimeType: "image/jpeg",
                    data: base64
                  }
                });
              }).catch(console.error);
            }
          }
          setTimeout(sendVideoFrame, 1000);
        };
        setTimeout(sendVideoFrame, 1000);

      } catch (err) {
        console.error("Setup error:", err);
        setIsConnecting(false);
      }
    }
    setup();

    return () => {
      active = false;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
      if (audioRecorderRef.current) {
        audioRecorderRef.current.stop();
      }
      if (audioPlayerRef.current) {
        audioPlayerRef.current.stop();
      }
      if (sessionRef.current) {
        sessionRef.current.then((s: any) => s.close()).catch(console.error);
      }
      if (captionTimeoutRef.current) {
        clearTimeout(captionTimeoutRef.current);
      }
    };
  }, [facingMode, targetLang]);

  const handleSnap = async () => {
    if (!videoRef.current || !canvasRef.current || isSnapping) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    setCapturedImage(dataUrl);
    setIsSnapping(true);
    setImageSummary(null);

    try {
      const ai = new GoogleGenAI({ apiKey: (import.meta as any).env?.VITE_GEMINI_API_KEY || process.env.GEMINI_API_KEY });
      const model = "gemini-3-flash-preview";
      const base64Data = dataUrl.split(',')[1];
      
      const response = await ai.models.generateContent({
        model,
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: "image/jpeg",
                data: base64Data
              }
            },
            {
              text: `Provide a very brief, concise translation summary of this image (e.g. menu, label, item) in ${targetLang}. 
              Focus on the most important information. 
              Output ONLY the summary text. No descriptions or conversational filler.`
            }
          ]
        },
        config: {
          temperature: 0.5
        }
      });
      
      setImageSummary(response.text || "Could not generate summary.");
    } catch (err) {
      console.error("Snap summary error:", err);
      setImageSummary("Error generating summary.");
    } finally {
      setIsSnapping(false);
    }
  };

  const handleSavePhoto = () => {
    if (!capturedImage) return;
    const link = document.createElement('a');
    link.href = capturedImage;
    link.download = `translation-snap-${new Date().getTime()}.jpg`;
    link.click();
  };

  return (
    <div className="fixed inset-0 bg-black">
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline 
        muted 
        className={`w-full h-full object-cover ${facingMode === 'user' ? 'scale-x-[-1]' : ''}`}
      />
      <canvas ref={canvasRef} className="hidden" />

      {/* Snap Overlay */}
      <AnimatePresence>
        {capturedImage && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 bg-black/80 backdrop-blur-sm flex flex-col pointer-events-auto"
          >
            <div className="flex justify-end p-6">
              <button 
                onClick={() => {
                  setCapturedImage(null);
                  setImageSummary(null);
                }}
                className="bg-white/10 hover:bg-white/20 p-3 rounded-full text-white transition-colors"
              >
                <X size={24} />
              </button>
            </div>

            <div className="flex-1 flex items-center justify-center p-4">
              <div className="relative max-w-full max-h-full aspect-auto rounded-3xl overflow-hidden shadow-2xl border border-white/10">
                <img src={capturedImage} alt="Captured" className="max-w-full max-h-[60vh] object-contain" />
                {isSnapping && (
                  <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                    <div className="w-8 h-8 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                  </div>
                )}
              </div>
            </div>

            <div className="p-8 pb-12 flex flex-col items-center gap-6">
              {imageSummary && (
                <motion.div 
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  className="bg-white text-stone-900 p-6 rounded-[2rem] shadow-2xl max-w-lg w-full"
                >
                  <p className="text-lg font-medium text-center leading-snug">
                    {imageSummary}
                  </p>
                </motion.div>
              )}

              <div className="flex gap-4">
                <button 
                  onClick={handleSavePhoto}
                  className="bg-white/10 hover:bg-white/20 text-white px-6 py-3 rounded-full flex items-center gap-2 font-medium transition-colors border border-white/10"
                >
                  <Download size={20} />
                  Save Photo
                </button>
                <button 
                  onClick={() => {
                    setCapturedImage(null);
                    setImageSummary(null);
                  }}
                  className="bg-indigo-600 hover:bg-indigo-700 text-white px-8 py-3 rounded-full font-medium transition-colors shadow-lg"
                >
                  Done
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="absolute inset-0 flex flex-col justify-between p-6 pointer-events-none">
        
        <div className="flex justify-between items-start pointer-events-auto w-full max-w-3xl mx-auto">
          <div className="bg-white/10 backdrop-blur-2xl border border-white/10 rounded-2xl px-5 py-3 flex items-center gap-3 text-white shadow-xl w-52 transition-all hover:bg-white/20">
            <Globe size={20} className="text-white/70" />
            <select 
              value={targetLang}
              onChange={(e) => {
                setTargetLang(e.target.value);
              }}
              className="bg-transparent text-white outline-none font-medium appearance-none w-full cursor-pointer"
            >
              <option value="English" className="text-black">English</option>
              <option value="Spanish" className="text-black">Spanish</option>
              <option value="French" className="text-black">French</option>
              <option value="German" className="text-black">German</option>
              <option value="Japanese" className="text-black">Japanese</option>
              <option value="Chinese" className="text-black">Chinese</option>
              <option value="Hindi" className="text-black">Hindi</option>
              <option value="Arabic" className="text-black">Arabic</option>
            </select>
          </div>

          <div className="flex gap-4">
            <button 
              onClick={() => setFacingMode(prev => prev === 'user' ? 'environment' : 'user')}
              className="bg-white/10 backdrop-blur-2xl border border-white/10 p-4 rounded-2xl text-white hover:bg-white/20 transition-all shadow-xl"
            >
              <FlipHorizontal size={22} />
            </button>
            <button 
              onClick={onStop}
              className="bg-white/10 backdrop-blur-2xl border border-white/10 p-4 rounded-2xl text-white hover:bg-white/20 transition-all shadow-xl"
            >
              <X size={22} />
            </button>
          </div>
        </div>

        <div className="flex flex-col items-center gap-6 w-full max-w-xl mx-auto mb-10 px-4 pointer-events-auto">
          {isConnecting && (
            <div className="bg-white/10 backdrop-blur-2xl border border-white/20 text-white/90 px-5 py-2 rounded-full text-xs font-semibold tracking-wider uppercase flex items-center gap-3 shadow-2xl">
              <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
              Connecting...
            </div>
          )}
          
          <AnimatePresence>
            {captions && (
              <motion.div 
                initial={{ opacity: 0, y: 10, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.98 }}
                className="bg-indigo-600/90 backdrop-blur-xl text-white px-6 py-4 rounded-[1.5rem] shadow-2xl border border-white/20 max-w-[90%]"
              >
                <p className="text-lg md:text-xl font-medium leading-tight text-center tracking-tight">
                  {captions}
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          <button 
            onClick={handleSnap}
            disabled={isSnapping}
            className="w-full bg-white/10 backdrop-blur-2xl border border-white/20 py-5 rounded-[2rem] text-white flex items-center justify-center gap-3 hover:bg-white/20 transition-all shadow-2xl disabled:opacity-50 group"
          >
            <Scan size={24} className="group-hover:scale-110 transition-transform" />
            <span className="text-lg font-medium tracking-tight">Snap to translate</span>
          </button>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [appState, setAppState] = useState<'intro' | 'ready' | 'live'>('intro');
  const [targetLang, setTargetLang] = useState('English');
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment');

  return (
    <div className="w-full h-screen bg-black overflow-hidden font-sans">
      {appState === 'intro' && <IntroScreen onComplete={() => setAppState('ready')} />}
      {appState === 'ready' && <ReadyScreen onStart={() => setAppState('live')} />}
      {appState === 'live' && (
        <LiveSession 
          targetLang={targetLang}
          setTargetLang={setTargetLang}
          facingMode={facingMode}
          setFacingMode={setFacingMode}
          onStop={() => setAppState('ready')} 
        />
      )}
    </div>
  );
}
