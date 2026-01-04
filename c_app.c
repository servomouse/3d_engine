// Build: gcc c_app.c -mwindows -lgdi32 -o main.exe

#include <windows.h>
#include <stdio.h>

#define ID_BTN_LINE 1
#define ID_BTN_CIRCLE 2
#define ID_BTN_CLEAR 3
#define ID_COMBOBOX 4
#define ID_TIMER_60FPS 100

// Global state
BOOL g_showLine = FALSE;
BOOL g_showCircle = FALSE;
int g_frameCount = 0;
int g_selectedShape = 0; 

// --- DRAWING FUNCTIONS ---

void RenderMainCanvas(HDC hdc, int width, int height) {
    // Fill background for main area only
    RECT canvasRect = {150, 0, width, height};
    FillRect(hdc, &canvasRect, (HBRUSH)GetStockObject(WHITE_BRUSH));

    HPEN bluePen = CreatePen(PS_SOLID, 2, RGB(0, 0, 255));
    HBRUSH redBrush = CreateSolidBrush(RGB(255, 0, 0));

    if (g_showLine || g_selectedShape == 1) {
        SelectObject(hdc, bluePen);
        MoveToEx(hdc, 250, 100, NULL); 
        LineTo(hdc, 550, 450);
    }

    if (g_showCircle || g_selectedShape == 2) {
        SelectObject(hdc, redBrush);
        Ellipse(hdc, 300, 150, 500, 350);
    }

    DeleteObject(bluePen);
    DeleteObject(redBrush);
}

void RenderSmallCanvas(HDC hdc) {
    // Define Small Canvas boundaries
    RECT smallBox = {15, 170, 135, 270};
    Rectangle(hdc, smallBox.left, smallBox.top, smallBox.right, smallBox.bottom);

    // Draw something unique to the small canvas
    // For now, let's draw a green square that grows/shrinks with the timer
    int pulse = (g_frameCount % 20);
    HBRUSH greenBrush = CreateSolidBrush(RGB(0, 180, 0));
    RECT pulseRect = {25 + pulse, 180 + pulse, 125 - pulse, 260 - pulse};
    
    FillRect(hdc, &pulseRect, greenBrush);
    DeleteObject(greenBrush);

    char countBuf[32];
    sprintf(countBuf, "Tick: %d", g_frameCount);
    TextOut(hdc, 15, 280, countBuf, (int)strlen(countBuf));
}

// --- WINDOW PROCEDURE ---

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CREATE: {
            CreateWindow("BUTTON", "Draw Line", WS_VISIBLE | WS_CHILD, 10, 10, 130, 30, hwnd, (HMENU)ID_BTN_LINE, NULL, NULL);
            CreateWindow("BUTTON", "Draw Circle", WS_VISIBLE | WS_CHILD, 10, 50, 130, 30, hwnd, (HMENU)ID_BTN_CIRCLE, NULL, NULL);
            CreateWindow("BUTTON", "Clear Canvas", WS_VISIBLE | WS_CHILD, 10, 90, 130, 30, hwnd, (HMENU)ID_BTN_CLEAR, NULL, NULL);
            
            HWND hCombo = CreateWindow("COMBOBOX", "", WS_VISIBLE | WS_CHILD | CBS_DROPDOWNLIST | WS_VSCROLL,
                                       10, 130, 130, 200, hwnd, (HMENU)ID_COMBOBOX, NULL, NULL);
            SendMessage(hCombo, CB_ADDSTRING, 0, (LPARAM)"Select Shape...");
            SendMessage(hCombo, CB_ADDSTRING, 0, (LPARAM)"Blue Line");
            SendMessage(hCombo, CB_ADDSTRING, 0, (LPARAM)"Red Circle");
            SendMessage(hCombo, CB_SETCURSEL, 0, 0);

            SetTimer(hwnd, ID_TIMER_60FPS, 16, NULL); // 16ms is roughly 60fps
            return 0;
        }

        case WM_TIMER: {
            g_frameCount++;
            // Invalidate everything to ensure both canvases update
            InvalidateRect(hwnd, NULL, FALSE);
            return 0;
        }

        case WM_ERASEBKGND: return 1;

        case WM_COMMAND: {
            if (LOWORD(wParam) == ID_COMBOBOX && HIWORD(wParam) == CBN_SELCHANGE) {
                g_selectedShape = (int)SendMessage((HWND)lParam, CB_GETCURSEL, 0, 0);
            }
            switch (LOWORD(wParam)) {
                case ID_BTN_LINE:   g_showLine = TRUE; break;
                case ID_BTN_CIRCLE: g_showCircle = TRUE; break;
                case ID_BTN_CLEAR:  g_showLine = FALSE; g_showCircle = FALSE; g_selectedShape = 0; break;
            }
            return 0;
        }

        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

            RECT rc;
            GetClientRect(hwnd, &rc);
            int width = rc.right - rc.left;
            int height = rc.bottom - rc.top;

            // --- Double Buffering ---
            HDC memDC = CreateCompatibleDC(hdc);
            HBITMAP memBitmap = CreateCompatibleBitmap(hdc, width, height);
            HBITMAP oldBitmap = (HBITMAP)SelectObject(memDC, memBitmap);

            // 1. Sidebar Background
            RECT sidebar = {0, 0, 150, height};
            FillRect(memDC, &sidebar, (HBRUSH)(COLOR_BTNFACE + 1));

            // 2. Call our new Drawing Functions
            RenderMainCanvas(memDC, width, height);
            RenderSmallCanvas(memDC);

            // 3. Final Copy
            BitBlt(hdc, 0, 0, width, height, memDC, 0, 0, SRCCOPY);

            // Cleanup
            SelectObject(memDC, oldBitmap);
            DeleteObject(memBitmap);
            DeleteDC(memDC);
            EndPaint(hwnd, &ps);
            return 0;
        }

        case WM_DESTROY:
            KillTimer(hwnd, ID_TIMER_60FPS);
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// WinMain remains the same as your code...
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    WNDCLASS wc = {0};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "FlickerFreeClass";
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, wc.lpszClassName, "Win32 C App - Refactored", 
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, CW_USEDEFAULT, CW_USEDEFAULT, 800, 600,
        NULL, NULL, hInstance, NULL);

    ShowWindow(hwnd, nShowCmd);
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return 0;
}