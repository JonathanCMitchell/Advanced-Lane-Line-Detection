# Conditionals to reset anchors automagically
# if len(good_lefts) < 20:
#     self.usePrevLeft = True
#     self.leftFound = False
# elif len(good_lefts) > 35:
#     self.leftFound = True
#     self.leftAnchor = left_max
# elif 35 > len(good_lefts) > 20:
#     self.leftFound = True
#
# if len(good_rights) < 20:
#     self.usePrevRight = True
#     self.rightFound = False
# elif len(good_rights) >= 35:
#     self.rightFound = True
#     self.rightAnchor = right_max
# elif 35 > len(good_rights) > 20:
#     self.rightFound = True


# # If the value found is far away from the previous left anchor do not append
# if self.leftFound == True and (abs(self.leftAnchor - left_max) < 20):
#     left_x_vals.append(left_max)
# else:
#     self.usePrevLeft = True
#
# # If the value found is far away from the previous right anchor do not append
# if self.rightFound == True and abs(self.rightAnchor - right_max) < 20:
#     right_x_vals.append(right_max)
# else:
#     self.usePrevRight = True


# If there is no left anchor or right anchor and we are at the first run
if self.first == True:
    for level in range(1, int(img_height / window_height)):
        window_start_vertical = img_height - (level + 1) * (window_height)
        window_end_vertical = img_height - (level * window_height)

        image_layer = np.sum(mask[window_start_vertical: window_end_vertical, :], axis=0)
        conv_signal = np.convolve(window, image_layer)

        offset = int(window_width / 2)
        l_min_index = int(max(left_max + offset - margin_width, 0))
        l_max_index = int(min(left_max + offset + margin_width, img_width))
        l_center = np.argmax(conv_signal[l_min_index: l_max_index]) + l_min_index - offset

        # look near the first centroid for the right centroids
        r_min_index = int(max(right_max + offset - margin_width, 0))
        r_max_index = int(min(right_max + offset + margin_width, img_width))
        r_center = np.argmax(conv_signal[r_min_index: r_max_index]) + r_min_index - offset

        # if abs(self.leftAnchor - l_center) < 20:
        #     left_x_vals.append(l_center)
        #
        # if abs(self.rightAnchor - r_center) < 20:
        #     right_x_vals.append(r_center)



left_centers = moving_average_scale(left_x_vals)
        right_centers = moving_average_scale(right_x_vals)

        if len(right_centers) > len(left_centers):
            # add left centers
            left_centers = add_recent_centers(len(right_centers), left_centers, self.left_recent_centers_averaged,
                                                      self.smooth_factor)
        elif len(left_centers) > len(right_centers):
            # add right centers
            right_centers = add_recent_centers(len(left_centers), right_centers, self.right_recent_centers_averaged,
                                                       self.smooth_factor)

        # print('left_centers after add_recent_centers: ', left_centers)
        #         print('right_centers after add_recent_centers: ', right_centers)
        #         print('self.right_recent_centers averaged: ', self.right_recent_centers_averaged)

        # once left and right centers are squared away!
        if len(left_centers) == len(right_centers):
            centroids = [[a, b] for a, b in zip(left_centers, right_centers)]
            for item in centroids:
                self.recent_centers.append(item)

        left_recent_centers = []
        right_recent_centers = []
        for i in reversed(range(len(self.recent_centers))):
            if len(left_recent_centers) < self.smooth_factor:
                left_recent_centers.append(self.recent_centers[i][0])
                right_recent_centers.append(self.recent_centers[i][1])

        # TODO: Fix: The recent centers just get reassigned each time, not added
        self.left_recent_centers_averaged = moving_average_scale(left_recent_centers)
        self.right_recent_centers_averaged = moving_average_scale(right_recent_centers)

        #         print('left_recent_centers: ', left_recent_centers)
        #         print('right_recent_centers: ', right_recent_centers)

        #         print('self.left_recent_centers_averaged: ', self.left_recent_centers_averaged)
        #         print('self.right_recent_centers_averaged: ', self.right_recent_centers_averaged)

        # TODO: If we have any flags for either left or right lane lines then roll:
        if self.usePrevLeft == True:
            print('usePrevLeft triggered')
            print('left_recent_centers averaged length: ', len(self.left_recent_centers_averaged))
            self.left_recent_centers_averaged = np.roll(self.left_recent_centers_averaged, len(left_recent_centers))
        if self.usePrevRight == True:
            print('usePrevRight triggered')
            print('right_recent_centers averaged length: ', len(self.right_recent_centers_averaged))
            print('self.right_recent_centers_averaged: ', self.right_recent_centers_averaged)
            self.right_recent_centers_averaged = np.roll(self.right_recent_centers_averaged, len(right_recent_centers))


        # TODO: Split this and send back recent_lefts_averaged and recent_rights_averaged
        # Then draw the lines individually
        # this way you can separate concerns for different lines
        if (len(self.left_recent_centers_averaged) & len(self.right_recent_centers_averaged)) < self.smooth_factor:
            self.recent_centers_averaged = [[a, b] for a, b in list(
                zip(self.left_recent_centers_averaged, self.right_recent_centers_averaged))]
        else:
            self.recent_centers_averaged = [[a, b] for a, b in
                                            list(zip(self.left_recent_centers_averaged[-self.smooth_factor:],
                                                     self.right_recent_centers_averaged[-self.smooth_factor:]))]

        print('RECENT_CENTERS_AVERAGED: ', self.recent_centers_averaged)



        return self.recent_centers_averaged