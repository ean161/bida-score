<?php
error_reporting(0);
$imagePath = 'data/cam.jpg';

$firstScore = true;
$balls = [];
$lastBall = [[1, 1]];

$side = "red";

$score = [
    "red" => 0,
    "blue" => 0
];

$move = [
    "white" => false,
    "red" => false,
    "orange" => false
];

$ps = [
    "white" => [0, 0],
    "red" => [0, 0],
    "orange" => [0, 0]
];

$lp = [
    "white" => [0, 0],
    "red" => [0, 0],
    "orange" => [0, 0]
];

$lastHit = time();
file_put_contents("data/state.txt", $side."|".$score["red"]."|".$score["blue"]);

while (true) {
    sleep(1);
    $allMatch = true;
    $detect = (detect());

    // print_r($detect);

    if (count($detect) != 3) {
        continue;
    }

    $text = "";
    file_put_contents("data/detect.txt", "");

    $newDetect = [];
    foreach ($detect as $obj) {
        $balls[] = [$obj["x"], $obj["orange"]];
        $text .= explode("_", $obj["class"])[0]."> ".$obj["x"].", ".$obj["orange"]."\n";

        $obj["class"] = explode("_", $obj["class"])[0];
        array_push($newDetect, $obj);

        $ps[$obj["class"]] = [$obj["x"], $obj["y"]];
    }
    $detect = $newDetect;

    file_put_contents("data/detect.txt", $text);

    foreach (["white", "red", "orange"] as $color) {
        if (!diff($ps[$color][0], $lp[$color][0]) || !diff($ps[$color][1], $lp[$color][1])) {
            $allMatch = false;
            $move[$color] = true;
            echo $color." is moving\n";
        }
    }

    if ($allMatch) {
        $iMove = 0;
        foreach (["white", "red", "orange"] as $color) {
            if ($move[$color] == true) {
                $iMove++;
            }
        }
        
        if ($iMove == 3) {
            if ($firstScore == true) {
                $firstScore = false;
                echo "Skipped first score\n";
                
                // Cập nhật trạng thái để không lặp lại skip
                foreach (["white", "red", "orange"] as $color) {
                    $move[$color] = false;
                }
                $lp = $ps;
        
                continue;
            }
        
            $lastHit = time();
            echo "+1 points\n";
            $score[$side]++;
            shell_exec("afplay './audio/attendance.mp3'  > /dev/null 2>&1 &");
        } else if ($iMove >= 1) {
            echo "Missed\n";
            $side = ($side == "red" ? "blue" : "red");
            echo $side." turn\n";
            shell_exec("afplay './audio/missed.mp3' > /dev/null 2>&1 &");
        }
        echo "All balls are static\n";
        foreach (["white", "red", "orange"] as $color) {
            $move[$color] = false;
        }
    }

    $lp = $ps;

    file_get_contents("https://ean.vn/bida/up.php?data=".$side."|".$score["red"]."|".$score["blue"]);
}

function detect() {
    $detect = [];
    $data = explode("\n", file_get_contents("./data/coor.txt"));

    foreach ($data as $line) {
        $part = explode("|", $line);

        if (!$part[0]) {
            continue;
        }

        $detect[] = [
            "class" => strtolower($part[0]),
            "x" => $part[1],
            "y" => $part[2]
        ];
    }

    return $detect;

    global $imagePath;
    $base64Image = base64_encode(file_get_contents($imagePath));
    $ch = curl_init('https://serverless.roboflow.com/rolling-ball/1?api_key=AmEsIxjknKiFXv8XGMs9&confidence=3&overlap=20');

    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $base64Image);
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Content-Type: application/x-www-form-urlencoded'
    ]);

    $response = curl_exec($ch);
    curl_close($ch);
    file_put_contents("data/log.txt", $response);
    return json_decode($response, true)["predictions"];
}

function diff($a, $b) {
    return abs($a - $b) <= 3;
}

// function maxConf($data) {
//     // Định nghĩa các nhóm theo prefix
//     $groupMap = [
//         'orange' => 'orange_orange',
//         'orange' => 'orange_orange',
//         // thêm các nhóm khác nếu cần, ví dụ:
//         'red' => 'red',
//         'white' => 'white'
//     ];

//     $result = [];

//     foreach ($data as $item) {
//         // Lấy prefix chữ cái từ class
//         preg_match('/^[a-zA-Z]+/', $item['class'], $matches);
//         $prefix = $matches[0] ?? $item['class'];

//         // Ánh xạ sang nhóm (nếu có trong map), không thì giữ nguyên prefix
//         $group = $groupMap[$prefix] ?? $prefix;

//         if (!isset($result[$group]) || $item['confidence'] > $result[$group]['confidence']) {
//             $result[$group] = $item;
//         }
//     }

//     return array_values($result);
// }
