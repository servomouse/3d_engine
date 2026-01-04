// Build: gcc c_app.c -mwindows -lgdi32 -o main.exe
#include <windows.h>
#include <stdio.h>

// Identifiers for UI elements
#define ID_BTN_LINE 1
#define ID_BTN_CIRCLE 2
#define ID_BTN_CLEAR 3
#define ID_COMBOBOX 4
#define ID_TIMER_60FPS 100  // Unique ID for our timer

// Global state (Simple flags for a pure C implementation)
BOOL g_showLine = FALSE;
BOOL g_showCircle = FALSE;
int g_frameCount = 0;
int g_selectedShape = 0; // 0 = None, 1 = Line, 2 = Circle

// Drawing Functions
void DrawMyLine(HDC hdc) {
    HPEN hPen = CreatePen(PS_SOLID, 2, RGB(0, 0, 255)); // Blue line
    SelectObject(hdc, hPen);
    MoveToEx(hdc, 200, 100, NULL);
    LineTo(hdc, 500, 400);
    DeleteObject(hPen);
}

void DrawMyCircle(HDC hdc) {
    HBRUSH hBrush = CreateSolidBrush(RGB(255, 0, 0)); // Red fill
    SelectObject(hdc, hBrush);
    Ellipse(hdc, 250, 150, 450, 350);
    DeleteObject(hBrush);
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CREATE: {
            // Sidebar Buttons
            CreateWindow("BUTTON", "Draw Line", WS_VISIBLE | WS_CHILD, 
                         10, 10, 130, 30, hwnd, (HMENU)ID_BTN_LINE, NULL, NULL);
            
            CreateWindow("BUTTON", "Draw Circle", WS_VISIBLE | WS_CHILD, 
                         10, 50, 130, 30, hwnd, (HMENU)ID_BTN_CIRCLE, NULL, NULL);
            
            CreateWindow("BUTTON", "Clear Canvas", WS_VISIBLE | WS_CHILD, 
                         10, 90, 130, 30, hwnd, (HMENU)ID_BTN_CLEAR, NULL, NULL);
            // Create the Dropdown
            HWND hCombo = CreateWindow("COMBOBOX", "", 
                        WS_VISIBLE | WS_CHILD | CBS_DROPDOWNLIST | WS_VSCROLL,
                        10, 130, 130, 200, hwnd, (HMENU)ID_COMBOBOX, NULL, NULL);

            // Add items to the dropdown
            SendMessage(hCombo, CB_ADDSTRING, 0, (LPARAM)"Select Shape...");
            SendMessage(hCombo, CB_ADDSTRING, 0, (LPARAM)"Blue Line");
            SendMessage(hCombo, CB_ADDSTRING, 0, (LPARAM)"Red Circle");

            // Set the default selection to index 0
            SendMessage(hCombo, CB_SETCURSEL, 0, 0);

            SetTimer(hwnd, ID_TIMER_60FPS, 40, NULL);
            return 0;
        }
        case WM_TIMER: {
            if (wParam == ID_TIMER_60FPS) {
                g_frameCount++;
                // Force the window to redraw every 16ms
                // FALSE means don't erase the whole background (prevents flickering)
                // FIX: Only invalidate the DRAWING area (x > 150), not the buttons!
                RECT canvasRect;
                GetClientRect(hwnd, &canvasRect);
                canvasRect.left = 151; // Start invalidation after the sidebar
                InvalidateRect(hwnd, &canvasRect, FALSE);
            }
            return 0;
        }
        case WM_ERASEBKGND:
            // Tell Windows we handled erasing the background. 
            // This is CRITICAL to stop flickering.
            return 1;

        case WM_COMMAND: {
            // Check if the message comes from our ComboBox
            if (LOWORD(wParam) == ID_COMBOBOX && HIWORD(wParam) == CBN_SELCHANGE) {
                // Get the index of the selected item
                int sel = (int)SendMessage((HWND)lParam, CB_GETCURSEL, 0, 0);
                g_selectedShape = sel; 
            }
            switch (LOWORD(wParam)) {
                case ID_BTN_LINE:   g_showLine = TRUE; break;
                case ID_BTN_CIRCLE: g_showCircle = TRUE; break;
                case ID_BTN_CLEAR:  g_showLine = FALSE; g_showCircle = FALSE; break;
            }
            // Invalidate triggers a redraw (WM_PAINT)
            InvalidateRect(hwnd, NULL, FALSE); 
            return 0;
        }

        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

            RECT rc;
            GetClientRect(hwnd, &rc);
            int width = rc.right - rc.left;
            int height = rc.bottom - rc.top;

            // Double Buffering
            HDC memDC = CreateCompatibleDC(hdc);
            HBITMAP memBitmap = CreateCompatibleBitmap(hdc, width, height);
            HBITMAP oldBitmap = (HBITMAP)SelectObject(memDC, memBitmap);

            // 1. Draw Sidebar Background (only if it needs painting)
            RECT sidebar = {0, 0, 150, height};
            FillRect(memDC, &sidebar, (HBRUSH)(COLOR_BTNFACE + 1));
            
            // 2. Draw Large Canvas Background
            RECT canvas = {150, 0, width, height};
            FillRect(memDC, &canvas, (HBRUSH)GetStockObject(WHITE_BRUSH));
            
            // 3. Draw "Small Canvas" Preview
            Rectangle(memDC, 15, 170, 135, 270);
            char countBuf[32];
            sprintf(countBuf, "Frame: %d", g_frameCount);
            TextOut(memDC, 15, 260, countBuf, (int)strlen(countBuf));

            if (g_selectedShape == 1) { // Blue Line
                HPEN pen = CreatePen(PS_SOLID, 2, RGB(0, 0, 255));
                SelectObject(memDC, pen);
                MoveToEx(memDC, 200, 200, NULL); LineTo(memDC, 600, 200);
                DeleteObject(pen);
            } 
            else if (g_selectedShape == 2) { // Red Circle
                HBRUSH brush = CreateSolidBrush(RGB(255, 0, 0));
                SelectObject(memDC, brush);
                Ellipse(memDC, 300, 100, 500, 300);
                DeleteObject(brush);
            }

            // 4. Draw the Shapes
            if (g_showLine) {
                HPEN pen = CreatePen(PS_SOLID, 2, RGB(0, 0, 255));
                SelectObject(memDC, pen);
                MoveToEx(memDC, 200, 100, NULL); LineTo(memDC, 500, 400);
                DeleteObject(pen);
            }
            if (g_showCircle) {
                HBRUSH brush = CreateSolidBrush(RGB(255, 0, 0));
                SelectObject(memDC, brush);
                Ellipse(memDC, 250, 150, 450, 350);
                DeleteObject(brush);
            }

            // 5. Blit to screen
            BitBlt(hdc, 0, 0, width, height, memDC, 0, 0, SRCCOPY);

            // Cleanup
            SelectObject(memDC, oldBitmap);
            DeleteObject(memBitmap);
            DeleteDC(memDC);

            EndPaint(hwnd, &ps);
            return 0;
        }

        case WM_DESTROY: {
            KillTimer(hwnd, ID_TIMER_60FPS);
            PostQuitMessage(0);
            return 0;
        }
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    WNDCLASS wc = {0};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "FlickerFreeClass";
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    
    RegisterClass(&wc);

    // KEY FIX: Added WS_CLIPCHILDREN to the window style
    HWND hwnd = CreateWindowEx(0, wc.lpszClassName, "Flicker-Free Pure C", 
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, 
        CW_USEDEFAULT, CW_USEDEFAULT, 800, 600,
        NULL, NULL, hInstance, NULL);

    ShowWindow(hwnd, nShowCmd);
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return 0;
}