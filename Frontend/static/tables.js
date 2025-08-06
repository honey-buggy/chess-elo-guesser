let options = {
    scales: {
        x: {
            ticks: {
                align: 'center'
            }
        },
        y: {
            min: 0,
        }
    },
    plugins: {
        title: {
            display: true,
            text: 'Loss vs thousand batches',
            font: {
                size: 20,
                weight: 'bold'
            },
            padding: {
                top: 5
            }
        },
        legend: {
            display: false
        }
    }
};


fetch('loss_tracking.csv')
    .then(response => response.text())
    .then(csvText => {
        const array = csvText
            .trim()
            .split(',')
            .map(Number);

        for (let i = 0; i < array.length; i++) {
            array[i] = {x: i, y: array[i]};
        }
        new Chart(document.getElementById('loss-chart'), {
            type: 'scatter',
            data: {
                labels: array,
                datasets: [{
                    data: array,
                    borderColor: '#57a6ff',
                }],
            },
            options: structuredClone(options),
        });
    });

const options1 = {
    scales: {
        x: {
            min: 400,
            max: 3100,
            ticks: {
                align: 'center'
            }
        },
        y: {
            min: 400,
            max: 3100,
        }
    },
    plugins: {
        subtitle:{
            display: true,
            text: '(Only 1000 data points shown)',
            font: {
                size: 16,
                weight: 'bold'
            },
            padding: {
                top: 5
            }
        },
        title: {
            display: true,
            text: 'Truth vs Predicted',
            font: {
                size: 20,
                weight: 'bold'
            },
            padding: {
                top: 5
            }
        },
        legend: {
            display: false
        }
    }
};

function parseCsvPoints(csv) {
    const pointsW = [];
    const pointsB = [];
    const rows = csv.trim().split('\n');

    for (const row of rows) {
        const values = row.split(',').map(Number);
        pointsW.push({x: values[0], y: values[1]});
        pointsB.push({x: values[2], y: values[3]});
    }

    return [pointsW, pointsB];
}

fetch('stats.csv')
    .then(response => response.text())
    .then(csvText => {
        const lineYX = [
            {x: 400, y:400},
            {x: 3100, y:3100},
        ];
        const points = parseCsvPoints(csvText)

        new Chart(document.getElementById('scatter-chart'), {
            type: 'scatter',
            data: {
                datasets: [{
                    label: "White",
                    data: points[0],
                    borderColor: '#57a6ff',
                }, {
                    label: "Black",
                    data: points[1],
                    borderColor: '#57a6ff',
                }, {
                    label: 'y = x',
                    data: lineYX,
                    borderColor: '#00cc00',
                    showLine: true,
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2,
                    order: 99
                }],
            },
            options: options1,
        });
    });

