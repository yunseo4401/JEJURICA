
from flask import Flask,render_template, request,send_file,Response
from werkzeug.utils import secure_filename
from io import StringIO
import os
import numpy as np
import pandas as pd
from kobart import * 
from functions import *

print('import 완료')



app=Flask(__name__)
@app.route("/",methods=['GET','POST'])
def index():
  if request.method=='GET':
     return render_template('index_1.html')
  elif request.method =='POST':
     if 'file' in request.files:
        f=request.files['file']
        path=os.path.join('C:\\Users\\diaky\\OneDrive\\Documents\\capstone_server'+ secure_filename(f.filename))
        f.save(path)
        print('저장완료')
        trans1=whisper(path)
        print('whisper완료')
        trans2=kobart(trans1,"C:\\Users\\diaky\\OneDrive\\Documents\\capstone_server\\epoch=02-val_loss=0.637.ckpt")
        print('kobart완료')
        trans3=mbart(trans2)
        print('mbart완료')
        tacotron2 = load_tactron2_model()
        hifi_gan = load_hifigan()
        mel_output, mel_length, alignment = tacotron2.encode_text(trans3)
        waveforms = hifi_gan.decode_batch(mel_output)
        output_stream = StringIO()
        with tempfile.NamedTemporaryFile(delete=False,suffix=".wav") as temp_wav:
            temp_wav_path=temp_wav.name
            torchaudio.save(temp_wav_path, waveforms.squeeze(1), 22050, format='wav', encoding='PCM_S')

             # 파일을 읽어와서 Response 객체 생성
            with open(temp_wav_path, 'rb') as temp_wav_file:
                response = Response(temp_wav_file.read(), mimetype='audio/x-wav', content_type='application/octet-stream')

        # Content-Disposition 헤더 설정
            response.headers['Content-Disposition'] = "attachment; filename=generated_audio.wav"
        return  response
     else:
        return render_template('index_1.html', file=None)
     
@app.route("/text",methods=['GET','POST'])
def translation():
  if request.method=='GET':
     return render_template('index_1.html')
  

  elif request.method =='POST':
     question = str(request.form['name'])
     trans2=kobart(question,"C:\\Users\\diaky\\OneDrive\\Documents\\capstone_server\\epoch=02-val_loss=0.637.ckpt")
     return render_template('index_1.html', kobart_answer=trans2)
   
if __name__ == '__main__':
   app.run(debug=True)