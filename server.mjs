import express from 'express';
import bodyParser from 'body-parser';
import { exec } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import iconv from 'iconv-lite';

const app = express();
const port = 3000;

// __dirname 대체 코드
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 정적 파일 제공 설정
app.use(express.static(path.join(__dirname)));

app.use(bodyParser.json());

app.post('/api/get-curriculum', (req, res) => {
    const userData = JSON.stringify(req.body);
    console.log('Received user data:', userData); // 클라이언트에서 받은 데이터 로그

    // Python 파일의 절대 경로를 사용
    const pythonScriptPath = path.join(__dirname, 'generate_roadmap.py');
    console.log('Start Python File');
    // python3 대신 python을 사용
    exec(`python ${pythonScriptPath}`, (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            return res.status(500).send(error);
        }

        // UTF-8 문자열을 파싱
        const result = JSON.parse(stdout.trim());

        
        console.log('Python script output:', result); // Python 스크립트 출력 로그
        res.json(result);
    });
});

// 기본 라우트 설정
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
