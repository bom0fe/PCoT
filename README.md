< PCoT 실행 방법 >


1. Node.js 서버 확인.
2. ngork.exe 실행하여 인증 및 local host 진행.
3. 인증과정: (1) ngrok config add-authtoken <key> 
            (2) ngrok http http://localhost:8080
   명령어 입력.
4. 'npm start'로 서버와 연결 및 웹 실행.
5. ngrok과 Node.js 서버가 잘 연결되었는지 확인하고, 뜬 웹링크로 접속
6. 웹 내에서 정보 작성. (사용자 정보는 submit과 동시에 스프레드시트에 저장) 
7. DB 내에서 정보를 api를 통해 불러와 generation 시작. (약 3분 이상 소요.)
8. 실습코드 다운로드를 통해 실습코드 확인 및 다운.
