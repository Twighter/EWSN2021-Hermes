clear;
path = './data/record.mat';
load(path);

[slotNum, ueNum]=size(throughputRecord);
frameSize = 10;
frameNum = slotNum / frameSize;
rbgNum = 6; %max(max(actionRecord));

figure(1);
heatmap(rbgNum - actionRecord');
title('Channel Selection');
xlabel('Slot');
ylabel('UE');
grid off;

throughput = zeros(frameNum, ueNum);
for i = 1:frameNum
    startIdx = (i - 1) * frameSize + 1;
    endIdx = i * frameSize;
    throughput(i, :) = sum(throughputRecord(startIdx : endIdx, :), 1);
end
figure(2);
heatmap(double(throughput)' * 100 / 1024 / 1024);
title('Throughput');
xlabel('Slot');
ylabel('UE');
grid off;

rbgCollision = zeros(frameNum, 1);
rbgIdle = zeros(frameNum, 1);
for i = 1:frameNum
    startIdx = (i - 1) * frameSize + 1;
    endIdx = i * frameSize;
    rbgCollision(i) = sum(rbgCollisionRecord(startIdx : endIdx));
    rbgIdle(i) = sum(rbgIdleRecord(startIdx : endIdx));
end
rbgUtil = frameSize * rbgNum - rbgCollision - rbgIdle;
figure(3);
plot(1:frameNum, rbgCollision / frameSize / rbgNum, ...
    1:frameNum, rbgIdle / frameSize / rbgNum, ...
    1:frameNum, rbgUtil / frameSize / rbgNum);
legend('collided', 'idle', 'utilized');
axis([1 frameNum 0 1]);
xlabel('Time (Frame)');
ylabel('Probability');