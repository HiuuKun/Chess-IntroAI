"""GUI pygame cho chess framework: menu chọn agent 2 bên + bàn cờ."""
import sys
import threading
import time
from pathlib import Path

import chess
import pygame

from agents import AGENT_FACTORIES
from core.game import Game

# Setting kích thước
ASSETS = Path(__file__).parent / "assets"
SQ = 80
BOARD = SQ * 8
PANEL_W = 260                      # Kích thước panel setting bên phải khi vào game
W, H = BOARD + PANEL_W, BOARD      # Chiều rộng = board + panel setting
FPS = 60
AI_DELAY = 1.0          

BG = (38, 36, 33)
CARD = (49, 46, 43)
CARD_HI = (66, 62, 57)
TEXT = (236, 235, 230)
MUTED = (160, 158, 150)
ACCENT = (129, 182, 76)
WARN = (232, 96, 80)
GOLD = (232, 190, 80)
BTN = (74, 70, 66)
BTN_HOVER = (96, 91, 85)
BTN_DIS = (50, 48, 45)

SQ_LIGHT = (222, 200, 165)
SQ_DARK = (150, 105, 70)

COLOR_VN = {chess.WHITE: "Trắng", chess.BLACK: "Đen"}
PIECE_ORDER = [chess.KING, chess.QUEEN, chess.BISHOP, chess.KNIGHT, chess.ROOK, chess.PAWN]
PROMO_CHOICES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

#--------Helper functions-----------

# Hàm lấy path của các ảnh assets
def asset(name: str) -> Path:
    p = ASSETS / name
    return p if p.exists() else Path(__file__).parent / name


def wrap(s: str, font, width: int) -> list[str]:
    lines, cur = [], ""
    for word in s.split():
        t = (cur + " " + word).strip()
        if font.size(t)[0] <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def runs(flags, gap):
    """Gom các chỉ số liên tiếp có flag True thành đoạn (bỏ qua khoảng trống <= gap)."""
    out, start, last = [], None, None
    for i, f in enumerate(flags):
        if not f:
            continue
        if start is None:
            start = i
        elif i - last > gap + 1:
            out.append((start, last))
            start = i
        last = i
    if start is not None:
        out.append((start, last))
    return out


def load_pieces() -> dict:
    """Cắt sprite sheet 6 cột x 2 hàng (hàng trên: trắng, dưới: đen;
    thứ tự K Q B N R P) bằng cách dò vùng có alpha, rồi scale về cùng tỉ lệ."""
    sheet = pygame.image.load(str(asset("pieces.png"))).convert_alpha()
    w, h = sheet.get_size()
    mask = pygame.mask.from_surface(sheet, 127)
    crops = {}
    for row, color in enumerate((chess.WHITE, chess.BLACK)):
        y0, y1 = (0, h // 2) if row == 0 else (h // 2, h)
        cols = runs([any(mask.get_at((x, y)) for y in range(y0, y1)) for x in range(w)], 3)
        if len(cols) != 6:                           # fallback: chia đều
            cw = w / 6
            cols = [(round(i * cw), round((i + 1) * cw) - 1) for i in range(6)]
        for pt, (x0, x1) in zip(PIECE_ORDER, cols):
            ys = [y for y in range(y0, y1)
                  if any(mask.get_at((x, y)) for x in range(x0, x1 + 1))]
            ya, yb = (ys[0], ys[-1]) if ys else (y0, y1 - 1)
            crops[(pt, color)] = sheet.subsurface((x0, ya, x1 - x0 + 1, yb - ya + 1)).copy()
    scale = SQ * 0.86 / max(max(c.get_size()) for c in crops.values())
    return {
        k: pygame.transform.smoothscale(
            c, (max(1, round(c.get_width() * scale)), max(1, round(c.get_height() * scale))))
        for k, c in crops.items()
    }


def fit(surf, size):
    w, h = surf.get_size()
    k = size / max(w, h)
    return pygame.transform.smoothscale(surf, (max(1, round(w * k)), max(1, round(h * k))))


class AIWorker:
    """Thread riêng để AI agent chạy song song với GUI rendering."""

    def __init__(self, game):
        self.game = game
        self.agent = game.current_agent
        self.board = game.board.copy()
        self.ply = len(game.move_history)
        self.result, self.error, self.done = None, None, False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            self.result = self.agent.choose_move(self.board)
        except Exception as e:          # noqa: BLE001
            self.error = e
        finally:
            self.done = True


class App:
    def __init__(self):
        # Tạo cửa sổ game
        pygame.init()
        pygame.display.set_caption("Chess")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()

        # Set các font chữ
        names = "segoeui,arial,dejavusans,notosans,helvetica,freesans"
        self.f_title = pygame.font.SysFont(names, 44, bold=True)
        self.f_big = pygame.font.SysFont(names, 34, bold=True)
        self.f_med = pygame.font.SysFont(names, 20)
        self.f_bold = pygame.font.SysFont(names, 20, bold=True)
        self.f_small = pygame.font.SysFont(names, 15)

        # Load bàn cờ và các quân cờ từ assets.
        img = pygame.image.load(str(asset("board.png"))).convert()
        self.board_img = pygame.transform.smoothscale(img, (BOARD, BOARD))
        self.pieces = load_pieces()
        self.icons = {c: fit(self.pieces[(chess.KING, c)], 38) for c in (chess.WHITE, chess.BLACK)}
        self._make_overlays()

        # Các biến để setup game state
        keys = list(AGENT_FACTORIES)        # Lấy toàn bộ các key đại diện agents thành một list
        self.choice = {chess.WHITE: keys[0], chess.BLACK: keys[1] if len(keys) > 1 else keys[0]}    # Mặc định WHITE = option1, BLACK = option2
        self.state = "menu"
        self.game: Game | None = None
        self.hit: list = []
        self._reset_view()


    def _make_overlays(self):
        "Chuẩn bị overlay để làm indicator hướng dẫn player."
        def solid(color):
            s = pygame.Surface((SQ, SQ), pygame.SRCALPHA) # Set kích thước bằng một ô, set trong suốt
            s.fill(color)
            return s
        self.ov_sel = solid((246, 246, 105, 150))   # Ô đang chọn
        self.ov_last = solid((205, 210, 106, 120))  # Last move
        self.ov_check = solid((235, 64, 52, 160))   # Cảnh báo checkmate
        self.ov_dot = pygame.Surface((SQ, SQ), pygame.SRCALPHA)     # Legal moves
        pygame.draw.circle(self.ov_dot, (20, 20, 20, 95), (SQ // 2, SQ // 2), SQ // 6) 
        self.ov_ring = pygame.Surface((SQ, SQ), pygame.SRCALPHA)    # Có thể ăn quân
        pygame.draw.circle(self.ov_ring, (20, 20, 20, 95), (SQ // 2, SQ // 2), SQ // 2 - 3, 7)

    def _reset_view(self):
        self.sel = None
        self.targets: dict[int, bool] = {}
        self.promo = None
        self.worker: AIWorker | None = None
        self.ai_error = None
        self.flipped = False
        self.popup_open = True
        self.prev_over = False
        self.outcome = None
        self.reason = ""
        self.san_rows = []
        self.last_ai_time = time.time()

    def start_game(self):
        if self.game:
            try:
                self.game.close()
            except Exception:           
                pass
        white = AGENT_FACTORIES[self.choice[chess.WHITE]]()     # Ở trên WHITE mặc định keys[0] nên white = agent đầu tiên trong dict
        black = AGENT_FACTORIES[self.choice[chess.BLACK]]()     # Tương tự với black
        self.game = Game(white, black)
        self._reset_view()
        # nếu chỉ có người chơi cầm quân đen thì quay bàn cờ cho dễ nhìn
        self.flipped = (not white.interactive) and black.interactive
        self.state = "game"
        self.refresh()

    def to_menu(self):
        self.worker = None
        self.state = "menu"

    # game state mỗi frame
    def refresh(self):
        g = self.game
        self.outcome = g.outcome()
        self.reason = g.end_reason() or ""
        over = self.outcome is not None
        if over and not self.prev_over:
            self.popup_open = True
        self.prev_over = over
        tmp, sans = chess.Board(), []
        # Lưu lại move history với cú pháp chuẩn san.
        for m in g.move_history:
            sans.append(tmp.san(m))
            tmp.push(m)
        self.san_rows = [
            (i // 2 + 1, sans[i], sans[i + 1] if i + 1 < len(sans) else "")
            for i in range(0, len(sans), 2)
        ]

    @property
    def last_move(self):
        "Lấy move gần nhất."
        h = self.game.move_history
        return h[-1] if h else None

    def apply_move(self, move) -> bool:
        "Apply move, xoaas các indicator và refresh bàn cờ"
        ok = self.game.make_move(move)
        self.sel, self.targets, self.promo = None, {}, None
        if ok:
            self.refresh()
        return ok

    def can_undo(self):
        g = self.game
        return (self.worker is None and bool(g.move_history)
                and any(a.interactive for a in g.agents.values()))

    def undo(self):
        if not self.can_undo():
            return
        g = self.game
        g.undo(1)
        while g.move_history and not g.current_agent.interactive:
            g.undo(1)                   # lùi tới lượt của người chơi
        self.sel, self.targets, self.promo, self.ai_error = None, {}, None, None
        self.last_ai_time = time.time()
        self.refresh()

    # ------------------------------------------------------------ tọa độ
    def sq_pos(self, sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if self.flipped:
            return (7 - f) * SQ, r * SQ
        return f * SQ, (7 - r) * SQ

    def pos_sq(self, pos):
        c, r = pos[0] // SQ, pos[1] // SQ
        if self.flipped:
            return chess.square(7 - c, r)
        return chess.square(c, 7 - r)

    # ------------------------------------------------------------ cập nhật AI
    def update(self):
        if self.state != "game":
            return
        g = self.game
        if self.worker:
            if not self.worker.done:
                return
            w, self.worker = self.worker, None
            if w.game is g and w.ply == len(g.move_history):
                if w.error:
                    self.ai_error = f"{w.agent.name} bị lỗi: {w.error}"
                elif not self.apply_move(w.result):
                    self.ai_error = f"{w.agent.name} trả về nước không hợp lệ: {w.result}"
                self.last_ai_time = time.time()
            return
        if (self.outcome or self.ai_error or self.promo
                or g.current_agent.interactive
                or time.time() - self.last_ai_time < AI_DELAY):
            return
        self.worker = AIWorker(g)

    # ------------------------------------------------------------ input
    def on_key(self, key):
        if key == pygame.K_ESCAPE:
            if self.state == "menu":
                self.quit()
            elif self.promo:
                self.promo = None
            elif self.outcome and self.popup_open:
                self.popup_open = False
            else:
                self.to_menu()
        elif self.state == "game":
            if key == pygame.K_z:
                self.undo()
            elif key == pygame.K_f:
                self.flipped = not self.flipped

    def on_click(self, pos):
        for rect, action in reversed(self.hit):
            if rect.collidepoint(pos):
                action()
                return
        if self.state != "game":
            return
        if self.promo:                       # bấm ra ngoài = hủy phong tốt
            self.promo = None
            return
        if self.outcome or pos[0] >= BOARD or pos[1] >= BOARD:
            return
        self.on_board_click(self.pos_sq(pos))

    def select(self, sq):
        g = self.game
        self.sel = sq
        self.targets = {m.to_square: g.board.is_capture(m) for m in g.legal_moves_from(sq)}

    def on_board_click(self, sq):
        g = self.game
        b = g.board
        if g.current_agent.interactive is False:
            return
        if self.sel is not None and sq in self.targets:
            piece = b.piece_at(self.sel)
            if piece.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                self.promo = (self.sel, sq)
            else:
                self.apply_move(g.build_move(self.sel, sq))
            return
        p = b.piece_at(sq)
        if p and p.color == b.turn and sq != self.sel:
            self.select(sq)
        else:
            self.sel, self.targets = None, {}

    def finish_promo(self, piece_type):
        frm, to = self.promo
        self.apply_move(chess.Move(frm, to, promotion=piece_type))

    # ------------------------------------------------------------ vẽ chung
    def text(self, s, font, color, pos, anchor="topleft"):
        img = font.render(s, True, color)
        r = img.get_rect(**{anchor: pos})
        self.screen.blit(img, r)
        return r

    def button(self, rect, label, action, enabled=True, selected=False,
               primary=False, font=None):
        rect = pygame.Rect(rect)
        hover = enabled and rect.collidepoint(pygame.mouse.get_pos())
        if not enabled:
            col = BTN_DIS
        elif selected or primary:
            col = (150, 205, 90) if hover else ACCENT
        else:
            col = BTN_HOVER if hover else BTN
        pygame.draw.rect(self.screen, col, rect, border_radius=8)
        fg = (20, 30, 10) if (enabled and (selected or primary)) else (TEXT if enabled else MUTED)
        self.text(label, font or self.f_med, fg, rect.center, "center")
        if enabled:
            self.hit.append((rect, action))

    def draw(self):
        self.hit.clear()
        if self.state == "menu":
            self.draw_menu()
        else:
            self.draw_board()
            self.draw_panel()
            if self.promo:
                self.draw_promo()
            elif self.outcome and self.popup_open:
                self.draw_popup()

    # ------------------------------------------------------------ menu
    def draw_menu(self):
        s = self.screen
        s.fill(BG)
        self.text("CHESS", self.f_title, TEXT, (W // 2, 62), "center")
        self.text("Chọn agent cho mỗi bên", self.f_med, MUTED, (W // 2, 102), "center")
        names = list(AGENT_FACTORIES)
        card_w, gap = 380, 40
        x0 = (W - 2 * card_w - gap) // 2
        for i, color in enumerate((chess.WHITE, chess.BLACK)):
            x = x0 + i * (card_w + gap)
            card = pygame.Rect(x, 130, card_w, 80 + len(names) * 54 + 6)
            pygame.draw.rect(s, CARD, card, border_radius=12)
            icon = fit(self.pieces[(chess.KING, color)], 52)
            s.blit(icon, icon.get_rect(center=(x + 44, card.y + 40)))
            self.text(f"Bên {COLOR_VN[color]}", self.f_big, TEXT, (x + 84, card.y + 40), "midleft")
            for j, name in enumerate(names):
                r = pygame.Rect(x + 20, card.y + 80 + j * 54, card_w - 40, 46)
                self.button(r, name, lambda c=color, n=name: self.choice.__setitem__(c, n),
                            selected=(self.choice[color] == name))
        self.button((W // 2 - 90, H - 168, 180, 38), "⇄  Đổi bên", self.swap_sides)
        self.button((W // 2 - 130, H - 112, 260, 58), "Bắt đầu", self.start_game,
                    primary=True, font=self.f_big)

    def swap_sides(self):
        c = self.choice
        c[chess.WHITE], c[chess.BLACK] = c[chess.BLACK], c[chess.WHITE]

    # ------------------------------------------------------------ bàn cờ
    def draw_board(self):
        s = self.screen
        g, b = self.game, self.game.board
        s.blit(self.board_img, (0, 0))

        if self.last_move:
            for sq in (self.last_move.from_square, self.last_move.to_square):
                s.blit(self.ov_last, self.sq_pos(sq))
        if b.is_check():
            s.blit(self.ov_check, self.sq_pos(b.king(b.turn)))
        if self.sel is not None:
            s.blit(self.ov_sel, self.sq_pos(self.sel))

        # nhãn tọa độ
        for c in range(8):
            f = 7 - c if self.flipped else c
            r = 7 if self.flipped else 0
            col = SQ_LIGHT if (f + r) % 2 == 0 else SQ_DARK
            self.text("abcdefgh"[f], self.f_small, col, (c * SQ + SQ - 4, BOARD - 3), "bottomright")
        for row in range(8):
            r = row if self.flipped else 7 - row
            f = 7 if self.flipped else 0
            col = SQ_LIGHT if (f + r) % 2 == 0 else SQ_DARK
            self.text(str(r + 1), self.f_small, col, (4, row * SQ + 3))

        # quân cờ
        for sq, p in b.piece_map().items():
            img = self.pieces[(p.piece_type, p.color)]
            x, y = self.sq_pos(sq)
            s.blit(img, img.get_rect(center=(x + SQ // 2, y + SQ // 2)))

        # nước đi khả dụng
        for sq, is_cap in self.targets.items():
            s.blit(self.ov_ring if is_cap else self.ov_dot, self.sq_pos(sq))

    def draw_promo(self):
        s = self.screen
        dim = pygame.Surface((BOARD, BOARD), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        s.blit(dim, (0, 0))
        color = self.game.turn
        pad = 14
        box = pygame.Rect(0, 0, 4 * SQ + 2 * pad, SQ + 2 * pad + 32)
        box.center = (BOARD // 2, BOARD // 2)
        pygame.draw.rect(s, CARD, box, border_radius=12)
        self.text("Phong tốt thành:", self.f_bold, TEXT, (box.centerx, box.y + 22), "center")
        self.hit.clear()
        mouse = pygame.mouse.get_pos()
        for i, pt in enumerate(PROMO_CHOICES):
            r = pygame.Rect(box.x + pad + i * SQ, box.y + pad + 32, SQ, SQ)
            pygame.draw.rect(s, (246, 246, 105) if r.collidepoint(mouse) else SQ_LIGHT, r,
                             border_radius=8)
            img = self.pieces[(pt, color)]
            s.blit(img, img.get_rect(center=r.center))
            self.hit.append((r, lambda p=pt: self.finish_promo(p)))

    # ------------------------------------------------------------ panel phải
    def status_text(self):
        g = self.game
        if self.outcome:
            o = self.outcome
            return "Hòa" if o.winner is None else f"{COLOR_VN[o.winner]} thắng"
        if self.ai_error:
            return self.ai_error
        if self.worker:
            return f"{g.current_agent.name} đang suy nghĩ..."
        s = f"Lượt của {COLOR_VN[g.turn]}"
        if g.board.is_check():
            s += " — Chiếu!"
        return s

    def draw_panel(self):
        s = self.screen
        g = self.game
        px = BOARD
        pygame.draw.rect(s, BG, (px, 0, PANEL_W, H))

        order = [chess.BLACK, chess.WHITE]
        if self.flipped:
            order.reverse()
        y = 14
        for color in order:
            active = not self.outcome and g.turn == color
            rect = pygame.Rect(px + 14, y, PANEL_W - 28, 56)
            pygame.draw.rect(s, CARD_HI if active else CARD, rect, border_radius=8)
            if active:
                pygame.draw.rect(s, ACCENT, rect, 2, border_radius=8)
            ic = self.icons[color]
            s.blit(ic, ic.get_rect(center=(rect.x + 28, rect.centery)))
            self.text(COLOR_VN[color], self.f_small, MUTED, (rect.x + 56, rect.y + 9))
            self.text(g.agents[color].name, self.f_bold, TEXT, (rect.x + 56, rect.y + 27))
            y += 64

        y = 150
        warn = bool(self.ai_error) or (not self.outcome and g.board.is_check())
        col = WARN if warn else (GOLD if self.outcome else TEXT)
        for line in wrap(self.status_text(), self.f_bold, PANEL_W - 28)[:3]:
            self.text(line, self.f_bold, col, (px + 14, y))
            y += 24

        self.text("Nước đi", self.f_small, MUTED, (px + 14, 236))
        for i, (n, w, bl) in enumerate(self.san_rows[-8:]):
            yy = 258 + i * 24
            self.text(f"{n}.", self.f_med, MUTED, (px + 14, yy))
            self.text(w, self.f_med, TEXT, (px + 56, yy))
            self.text(bl, self.f_med, TEXT, (px + 146, yy))

        bx, bw, bh = px + 14, PANEL_W - 28, 36
        by = H - 4 * 44 - 4
        self.button((bx, by, bw, bh), "Hoàn tác  (Z)", self.undo, enabled=self.can_undo())
        self.button((bx, by + 44, bw, bh), "Lật bàn  (F)",
                    lambda: setattr(self, "flipped", not self.flipped))
        self.button((bx, by + 88, bw, bh), "Xem kết quả",
                    lambda: setattr(self, "popup_open", True), enabled=bool(self.outcome))
        self.button((bx, by + 132, bw, bh), "Menu  (Esc)", self.to_menu)

    # ------------------------------------------------------------ pop-up kết thúc
    def draw_popup(self):
        s = self.screen
        g = self.game
        ov = pygame.Surface((W, H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 165))
        s.blit(ov, (0, 0))
        box = pygame.Rect(0, 0, 500, 300)
        box.center = (W // 2, H // 2)
        pygame.draw.rect(s, CARD, box, border_radius=14)
        pygame.draw.rect(s, CARD_HI, box, 2, border_radius=14)

        o = self.outcome
        if o.winner is None:
            title, sub, col = "Hòa", "Không bên nào thắng", GOLD
        else:
            title, col = f"{COLOR_VN[o.winner]} thắng", ACCENT
            sub = f"{COLOR_VN[not o.winner]} thua"
        self.text(title, self.f_title, col, (box.centerx, box.y + 48), "center")
        self.text(sub, self.f_med, TEXT, (box.centerx, box.y + 88), "center")
        self.text(f"Lý do: {self.reason}", self.f_bold, TEXT, (box.centerx, box.y + 124), "center")
        self.text(f"Kết quả {o.result()}   •   {len(g.move_history)} nước",
                  self.f_small, MUTED, (box.centerx, box.y + 154), "center")
        self.text(f"Trắng: {g.agents[chess.WHITE].name}   •   Đen: {g.agents[chess.BLACK].name}",
                  self.f_small, MUTED, (box.centerx, box.y + 176), "center")

        self.hit.clear()
        bw, gap = 140, 12
        x = box.x + (box.w - (3 * bw + 2 * gap)) // 2
        y = box.bottom - 66
        self.button((x, y, bw, 44), "Ván mới", self.start_game, primary=True)
        self.button((x + bw + gap, y, bw, 44), "Menu", self.to_menu)
        self.button((x + 2 * (bw + gap), y, bw, 44), "Xem bàn cờ",
                    lambda: setattr(self, "popup_open", False))

    # ------------------------------------------------------------ vòng lặp chính
    def quit(self):
        if self.game:
            try:
                self.game.close()
            except Exception:           
                pass
        pygame.quit()
        sys.exit()

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.quit()
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.on_click(e.pos)
                elif e.type == pygame.KEYDOWN:
                    self.on_key(e.key)
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    App().run()
