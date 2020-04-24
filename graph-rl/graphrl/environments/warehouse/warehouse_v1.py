import numpy as np
import gym
from pycolab import ascii_art, things
from pycolab.prefab_parts.sprites import MazeWalker

from graphrl.environments.pycolab_wrappers import load_art, PycolabMazeEnv

# Characters in the pycolab game

BACKGROUND_CHARACTER = ' '

WALL_CHARACTER = '+'

PLAYER_CHARACTER = 'A'
JUDGE_CHARACTER = chr(1)
MOVABLE_JUDGE_CHARACTER = chr(2)
FILLED_CHARACTER = 'X'


BUCKET_CHARACTERS = [chr(x) for x in range(ord('B'), ord('W') + 1)]
BOX_CHARACTERS = [chr(x) for x in range(256) if not chr(x).isspace() and chr(x) not in [BACKGROUND_CHARACTER, WALL_CHARACTER, PLAYER_CHARACTER, JUDGE_CHARACTER, MOVABLE_JUDGE_CHARACTER, FILLED_CHARACTER] + BUCKET_CHARACTERS]


ALL_CHARACTERS = [BACKGROUND_CHARACTER, WALL_CHARACTER, PLAYER_CHARACTER, JUDGE_CHARACTER, FILLED_CHARACTER]
ALL_CHARACTERS = ALL_CHARACTERS + BUCKET_CHARACTERS
ALL_CHARACTERS = ALL_CHARACTERS + BOX_CHARACTERS


FILL_BUCKET_REWARD = 3
STEP_PENALTY = 0.01
TERMINATE_PENALTY = 10


def maybe_initialize(the_plot):
    if 'filled_mask' not in the_plot:
        the_plot['filled_mask'] = {}
    if 'all_filled' not in the_plot:
        the_plot['all_filled'] = {}


class PlayerSprite(MazeWalker):
    def __init__(self, corner, position, character):
        super(PlayerSprite, self).__init__(corner, position, character, set(ALL_CHARACTERS) - set([character, BACKGROUND_CHARACTER]))

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is None:
            return
        maybe_initialize(the_plot)

        if actions == 0:
            self._north(board, the_plot)
        elif actions == 1:
            self._south(board, the_plot)
        elif actions == 2:
            self._west(board, the_plot)
        elif actions == 3:
            self._east(board, the_plot)


class BoxSprite(MazeWalker):
    def __init__(self, corner, position, character, bucket_to_boxes):
        impassable = set(ALL_CHARACTERS) - set([character, BACKGROUND_CHARACTER])

        self.bucket = None
        for bucket, boxes in bucket_to_boxes.items():
            if character in boxes and bucket in impassable:
                impassable.remove(bucket)
                self.bucket = bucket
        if self.bucket is None:
            raise ValueError('Could not find bucket.')

        super(BoxSprite, self).__init__(corner, position, character, impassable)

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is None:
            return
        maybe_initialize(the_plot)

        rows, cols = self.position
        if actions == 0:
            if layers[PLAYER_CHARACTER][rows + 1, cols]:
                self._north(board, the_plot)
        elif actions == 1:
            if layers[PLAYER_CHARACTER][rows - 1, cols]:
                self._south(board, the_plot)
        elif actions == 2:
            if layers[PLAYER_CHARACTER][rows, cols + 1]:
                self._west(board, the_plot)
        elif actions == 3:
            if layers[PLAYER_CHARACTER][rows, cols - 1]:
                self._east(board, the_plot)


class BucketDrape(things.Drape):
    def __init__(self, curtain, character, bucket_to_boxes):
        super(BucketDrape, self).__init__(curtain, character)
        self.last_buckets_filled = 0
        self.bucket_to_boxes = bucket_to_boxes

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is None:
            return
        maybe_initialize(the_plot)

        filled_mask = np.zeros_like(self.curtain)
        for box_char in self.bucket_to_boxes[self.character]:
            if box_char in all_things:
                filled_mask[all_things[box_char].position] = True
        filled_mask = np.logical_and(self.curtain, filled_mask)
        the_plot['filled_mask'][self.character] = filled_mask

        num_filled = np.sum(filled_mask)
        num_filled_change = num_filled - self.last_buckets_filled
        if num_filled_change != 0:
            the_plot.add_reward(FILL_BUCKET_REWARD * num_filled_change)
        if num_filled == self.curtain.sum():
            the_plot['all_filled'][self.character] = True
        else:
            the_plot['all_filled'][self.character] = False
        self.last_buckets_filled = num_filled


class FilledDrape(things.Drape):
    def __init__(self, curtain, character, buckets):
        super(FilledDrape, self).__init__(curtain, character)
        self.buckets = buckets

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is None:
            return
        maybe_initialize(the_plot)

        self.curtain.fill(False)
        for char in self.buckets:
            if char in all_things:
                filled_mask = the_plot['filled_mask'][char]
                np.logical_or(self.curtain, filled_mask, out=self.curtain)


class MovableJudgeDrape(things.Drape):
    def __init__(self, curtain, character):
        super(MovableJudgeDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        def is_src(x, y):
            if not (0 <= x < board.shape[0]):
                return False
            if not (0 <= y < board.shape[1]):
                return False
            char = chr(board[x, y])
            if char == BACKGROUND_CHARACTER:
                return True
            elif char == PLAYER_CHARACTER:
                return True
            else:
                return False

        def is_dst(x, y, bucket):
            if not (0 <= x < board.shape[0]):
                return False
            if not (0 <= y < board.shape[1]):
                return False
            char = chr(board[x, y])
            if char == BACKGROUND_CHARACTER:
                return True
            elif char == bucket:
                return True
            else:
                return False

        any_is_movable = False
        for k, sprite in all_things.items():
            if not isinstance(sprite, BoxSprite):
                continue
            row, col = sprite.position
            this_movable = (
                (is_src(row - 1, col) and is_dst(row + 1, col, sprite.bucket)) or
                (is_src(row + 1, col) and is_dst(row - 1, col, sprite.bucket)) or
                (is_src(row, col - 1) and is_dst(row, col + 1, sprite.bucket)) or
                (is_src(row, col + 1) and is_dst(row, col - 1, sprite.bucket))
            )
            any_is_movable = any_is_movable or this_movable
        the_plot['movable'] = any_is_movable


class JudgeDrape(things.Drape):
    def __init__(self, curtain, character, use_movable, use_frozen_penalty):
        super(JudgeDrape, self).__init__(curtain, character)
        self.use_movable = use_movable
        self.use_frozen_penalty = use_frozen_penalty

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is not None:
            maybe_initialize(the_plot)
            succeeded = True
            for bucket_all_filled in the_plot['all_filled'].values():
                if not bucket_all_filled:
                    succeeded = False
                    break

            if succeeded:
                the_plot.terminate_episode()
            elif self.use_movable and not the_plot['movable']:
                if self.use_frozen_penalty:
                    the_plot.add_reward(-TERMINATE_PENALTY)
                the_plot.terminate_episode()
            else:
                the_plot.add_reward(-STEP_PENALTY)


class WarehouseSuccessEnv(gym.Wrapper):
    def __init__(self, env, num_buckets):
        super(WarehouseSuccessEnv, self).__init__(env)
        self.num_buckets = num_buckets
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        self.collected = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if abs(FILL_BUCKET_REWARD - reward) < 2 * STEP_PENALTY:
            success = 1 / float(self.num_buckets)
        elif abs(FILL_BUCKET_REWARD + reward) < 2 * STEP_PENALTY:
            success = -1 / float(self.num_buckets)
        else:
            success = 0.
        info['episode_success'] = success
        return obs, reward, done, info


def make_warehouse_env(artfile, boxes, buckets, bucket_to_boxes, character_map={}, num_buckets=1, use_movable=False, use_frozen_penalty=False):
    art, art_characters, (height, width) = load_art(artfile)

    def make_pycolab_game():
        sprites = {}
        sprites[PLAYER_CHARACTER] = PlayerSprite

        game_boxes = [char for char in BOX_CHARACTERS if char in art_characters]
        game_buckets = [char for char in BUCKET_CHARACTERS if char in art_characters]

        for char in game_boxes:
            sprites[char] = ascii_art.Partial(BoxSprite, bucket_to_boxes=bucket_to_boxes)

        drapes = {}
        drapes[MOVABLE_JUDGE_CHARACTER] = MovableJudgeDrape
        drapes[JUDGE_CHARACTER] = ascii_art.Partial(JudgeDrape, use_movable=use_movable, use_frozen_penalty=use_frozen_penalty)
        drapes[FILLED_CHARACTER] = ascii_art.Partial(FilledDrape, buckets=list(bucket_to_boxes.keys()))
        for char in game_buckets:
            drapes[char] = ascii_art.Partial(BucketDrape, bucket_to_boxes=bucket_to_boxes)

        update_schedule = []
        update_schedule.append([MOVABLE_JUDGE_CHARACTER])
        update_schedule.append(game_boxes)
        update_schedule.append(game_buckets)
        update_schedule.append([PLAYER_CHARACTER, FILLED_CHARACTER, JUDGE_CHARACTER])

        return ascii_art.ascii_art_to_game(
            art,
            what_lies_beneath=BACKGROUND_CHARACTER,
            sprites=sprites,
            drapes=drapes,
            update_schedule=update_schedule
        )

    env = PycolabMazeEnv(make_game_function=make_pycolab_game,
                         num_actions=4,
                         height=height, width=width,
                         character_map=character_map)
    env = WarehouseSuccessEnv(env, num_buckets)

    return env
