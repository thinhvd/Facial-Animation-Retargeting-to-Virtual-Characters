var app = require('http').createServer(handler)
var io = require('socket.io')(app);
var fs = require('fs');

function handler(req, res) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    
    let url = req.url;
    if (url === '/') {
        url = '/model.html';
    }
    
    fs.readFile(__dirname + url,
        function (err, data) {
            if (err) {
                if (err.code === 'ENOENT') {
                    res.writeHead(404);
                    return res.end('File not found');
                }
                res.writeHead(500);
                return res.end('Error loading ' + url);
            }

            res.writeHead(200);
            res.end(data);
        });
}

io.of('/model').on('connection', (socket) => {
    console.log('a model client connected');

    socket.on('result_data', (result) => {
        if (result != 0) {
            socket.broadcast.emit('result_download', result);
        }
    });

    socket.on('disconnect', () => { console.log('a model client disconnected') });
});

io.of('/human').on('connection', (socket) => {
    console.log('a human client connected');
    socket.on('result_data', (result) => {
        if (result != 0) socket.broadcast.emit('result_download', result);
    });
    socket.on('disconnect', () => console.log('a human client disconnected'));
});

app.listen(6789, () => console.log('listening on http://127.0.0.1:6789/model.html'));
