clear all;
load bilayer_mm_zeroed.mat    

for i = 1:153
    for j = 1:23
        all_times_node_xpos = centroidslist_mm{i, 1};
        bilayer_x_positions(i, j) = all_times_node_xpos(j);

        all_times_node_ypos = centroidslist_mm{i, 2};
        bilayer_y_positions(i, j) = all_times_node_ypos(j);
    end
end

always_153 = [1, 2, 3, 20, 21, 22, 23];
top_40_leftmost = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
top_5_leftmost = [4];
for j = 1:23
    if ismember(j, always_153) 
        shift_idx(j) = 153;
    end
    
    if ismember(j, top_40_leftmost)
        [top_40_yvals,top_40_yindices] = maxk(bilayer_y_positions(:, j), 40, 1);
        
        xvals_to_look_at = bilayer_x_positions(top_40_yindices, j);
        min_xval = min(xvals_to_look_at);

        node_idx = find(bilayer_x_positions(:, j)==min_xval);

        shift_idx(j) = node_idx;
    end

    if ismember(j, top_5_leftmost)
        [top_5_yvals,top_5_yindices] = maxk(bilayer_y_positions(:, j), 5, 1);
        
        xvals_to_look_at = bilayer_x_positions(top_5_yindices, j);
        min_xval = min(xvals_to_look_at);

        node_idx = find(bilayer_x_positions(:, j)==min_xval);

        shift_idx(j) = node_idx;
    end

end

for j = 1:23
    curr_shift_idx = shift_idx(j);
    
    bilayer_x_positions_transformed(:, j) = bilayer_x_positions(:,j) - bilayer_x_positions(curr_shift_idx,j);
    bilayer_y_positions_transformed(:, j) = bilayer_y_positions(:,j) - bilayer_y_positions(curr_shift_idx,j);

end




% for i = 1:23
%     figure()
%     plot(bilayer_x_positions(:, i), bilayer_y_positions(:, i), 'ro')
%     hold on
%     plot(bilayer_x_positions(153, i), bilayer_y_positions(153, i), 'bo')
%     plot(bilayer_x_positions_transformed(:, i), bilayer_y_positions_transformed(:, i), 'ko')
%     plot(0, 0, 'go')
% end

%% rotate matrices
theta = pi/2;
rot_matrix = [cos(theta) -sin(theta); sin(theta), cos(theta)];
for i = 1:23
    curr_pos(:, 1) = bilayer_x_positions_transformed(:, i);
    curr_pos(:, 2) = bilayer_y_positions_transformed(:, i);
    
    new_pos = rot_matrix*curr_pos';

    bilayer_x_pos_transformed(:, i) = new_pos(1, :);
    bilayer_y_pos_transformed(:, i) = new_pos(2, :);
end

for i = 1:23
    figure()
    plot(bilayer_x_positions(:, i), bilayer_y_positions(:, i), 'ro')
    hold on

    plot(bilayer_x_positions_transformed(:, i), bilayer_y_positions_transformed(:, i), 'ko')
    plot(bilayer_x_pos_transformed(:, i), bilayer_y_pos_transformed(:, i), 'bo')
    plot(0, 0, 'go')
end